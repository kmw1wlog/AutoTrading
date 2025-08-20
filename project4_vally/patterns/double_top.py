import numpy as np
import pandas as pd

# -----------------------------
# Pivot 고점/저점 탐지
# -----------------------------
def find_pivots(high: pd.Series, low: pd.Series, nL=3, nR=3):
    H = pd.Series(False, index=high.index)
    L = pd.Series(False, index=low.index)
    for i in range(nL, len(high)-nR):
        if high.iloc[i] == high.iloc[i-nL:i+nR+1].max():
            H.iloc[i] = True
        if low.iloc[i] == low.iloc[i-nL:i+nR+1].min():
            L.iloc[i] = True
    return H, L

# -----------------------------
# ATR (리스크 관리)
# -----------------------------
def atr(df: pd.DataFrame, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# 이중천장/이중바닥 탐지 + 돌파 시그널
# -----------------------------
def detect_double_top_bottom(
    df: pd.DataFrame,
    pivot_left=3, pivot_right=3,   # 피벗 민감도
    lookback=300,                  # 탐지 최대 과거 구간
    peak_diff_max=0.01,            # 두 꼭대기(또는 두 바닥) 상대차 허용치(1% 이내 등)
    min_separation=5,              # 첫 꼭대기~둘째 꼭대기(또는 바닥) 최소 간격(봉수)
    max_separation=60,             # 최대 간격(봉수)
    min_pullback=0.015,            # 첫 꼭대기→넥라인(중간저점)까지 하락폭 최소치(1.5% 등)
    buffer=0.001,                  # 넥라인 돌파 판정 버퍼(0.1%)
    atr_period=14,
    rr_measure=1.0,                # 측정치 타깃 배수(머리-넥라인 높이 × 배수)
    stop_atr_mult=1.0,             # ATR 여유폭
    hold_bars=80,                  # 타임아웃 청산
    enable_double_top=True,
    enable_double_bottom=True
):
    """
    반환 DataFrame:
      ['pattern','neckline','p1','p2','mid','entry_short','entry_long',
       'exit_short','exit_long','stop_price','tp_price']
    pattern: 'DoubleTop' 또는 'DoubleBottom'
    p1,p2,mid: 첫/둘째 꼭대기(또는 바닥) 값과 중간 피벗값(넥라인 위치)
    """
    df = df.copy()
    idx = df.index
    n = len(df)

    Hflag, Lflag = find_pivots(df['High'], df['Low'], nL=pivot_left, nR=pivot_right)
    pivH = [(i, df['High'].iloc[i]) for i in range(n) if Hflag.iloc[i]]
    pivL = [(i, df['Low' ].iloc[i]) for i in range(n) if Lflag.iloc[i]]

    df['ATR'] = atr(df, period=atr_period)

    pattern     = np.array([None]*n, dtype=object)
    neckline    = np.full(n, np.nan)
    p1v = np.full(n, np.nan); p2v = np.full(n, np.nan); midv = np.full(n, np.nan)

    entry_short = np.zeros(n, dtype=bool)
    entry_long  = np.zeros(n, dtype=bool)
    exit_short  = np.zeros(n, dtype=bool)
    exit_long   = np.zeros(n, dtype=bool)
    stop_price  = np.full(n, np.nan)
    tp_price    = np.full(n, np.nan)

    start = max(0, n - lookback)

    # -------- Double Top (약세) --------
    if enable_double_top and len(pivH) >= 2 and len(pivL) >= 1:
        for i in range(len(pivH)-1):
            b1, v1 = pivH[i]
            # 분리 조건
            for j in range(i+1, len(pivH)):
                b2, v2 = pivH[j]
                if b2 - b1 < min_separation or b2 - b1 > max_separation:
                    continue
                if b1 < start:
                    continue
                # 두 꼭대기 유사성
                if abs(v2 - v1) / max(v1, 1e-12) > peak_diff_max:
                    continue
                # 중간 저점(넥라인 후보) 찾기: b1~b2 사이 최저 저점 피벗
                mids = [p for p in pivL if b1 < p[0] < b2]
                if not mids:
                    continue
                mid_bar, mid_low = min(mids, key=lambda x: x[1])  # 최저 저점
                # 최소 되돌림 폭(첫 꼭대기→중간저점)
                if (v1 - mid_low) / max(v1, 1e-12) < min_pullback:
                    continue

                # 우봉(b2) 이후 넥라인(mid_low) 하향 이탈 확인
                # 패턴 라벨은 b2 시점에 표기
                neckline[b2] = mid_low
                p1v[b2] = v1; p2v[b2] = v2; midv[b2] = mid_low
                pattern[b2] = 'DoubleTop'

                # 엔트리: b2 이후, 종가 < 넥라인*(1-buffer)
                for k in range(b2+1, min(b2+hold_bars, n-1)+1):
                    neck_k = mid_low  # 수평 넥라인
                    if df['Close'].iloc[k] < neck_k * (1 - buffer):
                        entry_short[k] = True
                        entry = df['Close'].iloc[k]
                        atr_e = df['ATR'].iloc[k]
                        # 손절: 두 번째 꼭대기 위 + ATR 여유 또는 넥라인 위 중 보수적(더 높은 값)
                        sj = max(v2, neck_k) + stop_atr_mult * atr_e
                        # 목표: 측정치(꼭대기-넥라인) × rr_measure
                        measure = (max(v1, v2) - mid_low) * rr_measure
                        tj = entry - measure
                        end_k = min(k + hold_bars, n-1)
                        for t in range(k+1, end_k+1):
                            if df['High'].iloc[t] >= sj:
                                exit_short[t] = True; stop_price[t] = sj; break
                            if df['Low'].iloc[t] <= tj:
                                exit_short[t] = True;  tp_price[t] = tj;  break
                        else:
                            exit_short[end_k] = True
                        break  # 한 패턴당 한 번만 진입

    # -------- Double Bottom (강세) --------
    if enable_double_bottom and len(pivL) >= 2 and len(pivH) >= 1:
        for i in range(len(pivL)-1):
            b1, v1 = pivL[i]
            for j in range(i+1, len(pivL)):
                b2, v2 = pivL[j]
                if b2 - b1 < min_separation or b2 - b1 > max_separation:
                    continue
                if b1 < start:
                    continue
                if abs(v2 - v1) / max(v1, 1e-12) > peak_diff_max:
                    continue
                # 중간 고점(넥라인 후보): b1~b2 사이 최고 고점 피벗
                mids = [p for p in pivH if b1 < p[0] < b2]
                if not mids:
                    continue
                mid_bar, mid_high = max(mids, key=lambda x: x[1])
                # 최소 되돌림 폭(첫 바닥→중간고점)
                if (mid_high - v1) / max(mid_high, 1e-12) < min_pullback:
                    continue

                neckline[b2] = mid_high
                p1v[b2] = v1; p2v[b2] = v2; midv[b2] = mid_high
                pattern[b2] = 'DoubleBottom'

                for k in range(b2+1, min(b2+hold_bars, n-1)+1):
                    neck_k = mid_high
                    if df['Close'].iloc[k] > neck_k * (1 + buffer):
                        entry_long[k] = True
                        entry = df['Close'].iloc[k]
                        atr_e = df['ATR'].iloc[k]
                        # 손절: 두 번째 바닥 아래 + ATR 여유 또는 넥라인 아래 중 보수적(더 낮은 값)
                        sj = min(v2, neck_k) - stop_atr_mult * atr_e
                        measure = (mid_high - min(v1, v2)) * rr_measure
                        tj = entry + measure
                        end_k = min(k + hold_bars, n-1)
                        for t in range(k+1, end_k+1):
                            if df['Low'].iloc[t] <= sj:
                                exit_long[t] = True; stop_price[t] = sj; break
                            if df['High'].iloc[t] >= tj:
                                exit_long[t] = True;  tp_price[t] = tj;  break
                        else:
                            exit_long[end_k] = True
                        break

    out = pd.DataFrame({
        'pattern': pd.Series(pattern, index=idx, dtype="object"),
        'neckline': neckline,
        'p1': p1v, 'p2': p2v, 'mid': midv,
        'entry_short': entry_short, 'entry_long': entry_long,
        'exit_short':  exit_short,  'exit_long':  exit_long,
        'stop_price':  stop_price,  'tp_price':   tp_price
    }, index=idx)

    return out

# -----------------------------
# 예시
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')  # ['Open','High','Low','Close']
    # sig = detect_double_top_bottom(df,
    #     peak_diff_max=0.01, min_separation=5, max_separation=60, min_pullback=0.015)
    # print(sig.tail(60))
    pass
