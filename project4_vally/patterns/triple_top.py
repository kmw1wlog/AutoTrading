import numpy as np
import pandas as pd

# -----------------------------------
# Pivot 고점/저점 탐지
# -----------------------------------
def find_pivots(high: pd.Series, low: pd.Series, nL=3, nR=3):
    H = pd.Series(False, index=high.index)
    L = pd.Series(False, index=low.index)
    for i in range(nL, len(high)-nR):
        if high.iloc[i] == high.iloc[i-nL:i+nR+1].max():
            H.iloc[i] = True
        if low.iloc[i]  == low.iloc[i-nL:i+nR+1].min():
            L.iloc[i] = True
    return H, L

# -----------------------------------
# ATR
# -----------------------------------
def atr(df: pd.DataFrame, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------------
# 삼중천장/삼중바닥 탐지 + 돌파 시그널
# -----------------------------------
def detect_triple_top_bottom(
    df: pd.DataFrame,
    pivot_left=3, pivot_right=3,     # 피벗 민감도
    lookback=400,                    # 탐지 최대 과거 구간
    peak_diff_max=0.012,             # 세 꼭대기(또는 세 바닥) 높이 유사 허용(1.2% 등)
    min_separation=5,                # 인접 피벗 최소 간격(봉수)
    max_separation=90,               # 첫~셋째 최대 간격(봉수)
    min_pullback=0.015,              # 첫 꼭대기→중간저점(또는 바닥→중간고점) 최소 되돌림
    buffer=0.001,                    # 돌파 버퍼(0.1%)
    atr_period=14,
    rr_measure=1.0,                  # 측정치 타깃 배수
    stop_atr_mult=1.0,               # 손절 ATR 여유
    hold_bars=100,                   # 타임아웃 청산
    enable_triple_top=True,
    enable_triple_bottom=True
):
    """
    반환 DataFrame:
      columns = [
        'pattern','neckline','p1','p2','p3','v12','v23',
        'entry_short','entry_long','exit_short','exit_long',
        'stop_price','tp_price'
      ]
    pattern: 'TripleTop' 또는 'TripleBottom'
    p1,p2,p3: 세 꼭대기(또는 바닥) 값
    v12,v23: 두 중간 피벗(넥라인 형성에 사용)
    """
    df = df.copy()
    idx = df.index
    n = len(df)

    Hflag, Lflag = find_pivots(df['High'], df['Low'], nL=pivot_left, nR=pivot_right)
    pivH = [(i, df['High'].iloc[i]) for i in range(n) if Hflag.iloc[i]]
    pivL = [(i, df['Low' ].iloc[i]) for i in range(n) if Lflag.iloc[i]]

    df['ATR'] = atr(df, period=atr_period)

    pattern      = np.array([None]*n, dtype=object)
    neckline     = np.full(n, np.nan)
    p1v = np.full(n, np.nan); p2v = np.full(n, np.nan); p3v = np.full(n, np.nan)
    v12v = np.full(n, np.nan); v23v = np.full(n, np.nan)

    entry_short = np.zeros(n, dtype=bool)
    entry_long  = np.zeros(n, dtype=bool)
    exit_short  = np.zeros(n, dtype=bool)
    exit_long   = np.zeros(n, dtype=bool)
    stop_price  = np.full(n, np.nan)
    tp_price    = np.full(n, np.nan)

    start = max(0, n - lookback)

    # ----------------- Triple Top (약세) -----------------
    if enable_triple_top and len(pivH) >= 3 and len(pivL) >= 2:
        Hbars = [b for b,_ in pivH]
        for a in range(len(Hbars)-2):
            i1 = Hbars[a]
            if i1 < start: 
                continue
            for b in range(a+1, len(Hbars)-1):
                i2 = Hbars[b]
                if i2 - i1 < min_separation:
                    continue
                for c in range(b+1, len(Hbars)):
                    i3 = Hbars[c]
                    if i3 - i2 < min_separation: 
                        continue
                    if i3 - i1 > max_separation:
                        continue

                    v1 = df['High'].iloc[i1]
                    v2 = df['High'].iloc[i2]
                    v3 = df['High'].iloc[i3]
                    v_mean = np.mean([v1, v2, v3])
                    # 세 꼭대기 유사성(각각 평균과의 상대 오차)
                    if any(abs(v - v_mean)/v_mean > peak_diff_max for v in [v1, v2, v3]):
                        continue

                    # 중간 저점(두 구간): i1~i2, i2~i3
                    mids12 = [p for p in pivL if i1 < p[0] < i2]
                    mids23 = [p for p in pivL if i2 < p[0] < i3]
                    if not mids12 or not mids23:
                        continue
                    mid12_bar, mid12_low = min(mids12, key=lambda x: x[1])
                    mid23_bar, mid23_low = min(mids23, key=lambda x: x[1])

                    # 최소 되돌림(첫 꼭대기→첫 중간저점)
                    if (v1 - mid12_low)/max(v1,1e-12) < min_pullback:
                        continue

                    # 넥라인: 보수적으로 두 중간저점 중 더 낮은 값(수평 가정)
                    neck = min(mid12_low, mid23_low)

                    # 패턴 확정 시점: 세 번째 꼭대기(i3)
                    neckline[i3] = neck
                    p1v[i3], p2v[i3], p3v[i3] = v1, v2, v3
                    v12v[i3], v23v[i3] = mid12_low, mid23_low
                    pattern[i3] = 'TripleTop'

                    # i3 이후 넥라인 하향 이탈 시 숏
                    for k in range(i3+1, min(i3+hold_bars, n-1)+1):
                        if df['Close'].iloc[k] < neck * (1 - buffer):
                            entry_short[k] = True
                            entry = df['Close'].iloc[k]
                            atr_e = df['ATR'].iloc[k]
                            # 손절: 세 번째 꼭대기 위 또는 넥라인 위 + ATR 여유 중 더 높은 쪽
                            sj = max(v3, neck) + stop_atr_mult*atr_e
                            # 측정치: 평균 꼭대기 - 넥라인
                            measure = (v_mean - neck) * rr_measure
                            tj = entry - measure
                            end_k = min(k + hold_bars, n-1)
                            for t in range(k+1, end_k+1):
                                if df['High'].iloc[t] >= sj:
                                    exit_short[t] = True; stop_price[t] = sj; break
                                if df['Low'].iloc[t] <= tj:
                                    exit_short[t] = True;  tp_price[t] = tj;  break
                            else:
                                exit_short[end_k] = True
                            break  # 한 패턴 1회 진입
    # ----------------- Triple Bottom (강세) -----------------
    if enable_triple_bottom and len(pivL) >= 3 and len(pivH) >= 2:
        Lbars = [b for b,_ in pivL]
        for a in range(len(Lbars)-2):
            i1 = Lbars[a]
            if i1 < start:
                continue
            for b in range(a+1, len(Lbars)-1):
                i2 = Lbars[b]
                if i2 - i1 < min_separation:
                    continue
                for c in range(b+1, len(Lbars)):
                    i3 = Lbars[c]
                    if i3 - i2 < min_separation:
                        continue
                    if i3 - i1 > max_separation:
                        continue

                    v1 = df['Low'].iloc[i1]
                    v2 = df['Low'].iloc[i2]
                    v3 = df['Low'].iloc[i3]
                    v_mean = np.mean([v1, v2, v3])
                    if any(abs(v - v_mean)/v_mean > peak_diff_max for v in [v1, v2, v3]):
                        continue

                    mids12 = [p for p in pivH if i1 < p[0] < i2]
                    mids23 = [p for p in pivH if i2 < p[0] < i3]
                    if not mids12 or not mids23:
                        continue
                    mid12_bar, mid12_high = max(mids12, key=lambda x: x[1])
                    mid23_bar, mid23_high = max(mids23, key=lambda x: x[1])

                    if (mid12_high - v1)/max(mid12_high,1e-12) < min_pullback:
                        continue

                    neck = max(mid12_high, mid23_high)

                    neckline[i3] = neck
                    p1v[i3], p2v[i3], p3v[i3] = v1, v2, v3
                    v12v[i3], v23v[i3] = mid12_high, mid23_high
                    pattern[i3] = 'TripleBottom'

                    for k in range(i3+1, min(i3+hold_bars, n-1)+1):
                        if df['Close'].iloc[k] > neck * (1 + buffer):
                            entry_long[k] = True
                            entry = df['Close'].iloc[k]
                            atr_e = df['ATR'].iloc[k]
                            sj = min(v3, neck) - stop_atr_mult*atr_e   # 더 낮은 쪽 - ATR
                            measure = (neck - v_mean) * rr_measure
                            tj = entry + measure
                            end_k = min(k + hold_bars, n-1)
                            for t in range(k+1, end_k+1):
                                if df['Low'].iloc[t] <= sj:
                                    exit_long[t]  = True; stop_price[t] = sj; break
                                if df['High'].iloc[t] >= tj:
                                    exit_long[t]  = True;  tp_price[t] = tj;  break
                            else:
                                exit_long[end_k] = True
                            break

    out = pd.DataFrame({
        'pattern': pd.Series(pattern, index=idx, dtype='object'),
        'neckline': neckline,
        'p1': p1v, 'p2': p2v, 'p3': p3v,
        'v12': v12v, 'v23': v23v,
        'entry_short': entry_short, 'entry_long': entry_long,
        'exit_short':  exit_short,  'exit_long':  exit_long,
        'stop_price':  stop_price,  'tp_price':   tp_price
    }, index=idx)
    return out

# -----------------------------------
# 예시
# -----------------------------------
if __name__ == "__main__":
    # df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')  # ['Open','High','Low','Close']
    # sig = detect_triple_top_bottom(df,
    #     peak_diff_max=0.012, min_separation=5, max_separation=90, min_pullback=0.015)
    # print(sig.tail(80))
    pass
