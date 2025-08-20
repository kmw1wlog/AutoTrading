import numpy as np
import pandas as pd

# -----------------------------
# Pivot 고점/저점
# -----------------------------
def find_pivots(high: pd.Series, low: pd.Series, nL=3, nR=3):
    H = pd.Series(False, index=high.index)
    L = pd.Series(False, index=low.index)
    for i in range(nL, len(high)-nR):
        if high.iloc[i] == high.iloc[i-nL:i+nR+1].max():
            H.iloc[i] = True
        if low.iloc[i]  == low.iloc[i-nL:i+nR+1].min():
            L.iloc[i] = True
    return H, L

# 직선 값(x0,y0)-(x1,y1)에서 xq에 대한 보간/외삽
def line_y(x0, y0, x1, y1, xq):
    if x1 == x0: 
        return y0
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

# 선분 기울기
def slope(x0, y0, x1, y1):
    return (y1 - y0) / (x1 - x0) if x1 != x0 else np.inf

# -----------------------------
# ATR
# -----------------------------
def atr(df: pd.DataFrame, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# Wolfe Wave 탐지 + 엔트리/청산
# -----------------------------
def detect_wolfe_wave(
    df: pd.DataFrame,
    pivot_left=3, pivot_right=3,    # 피벗 민감도
    lookback=600,                   # 최대 탐지 과거 구간
    min_sep=3,                      # 인접 피벗 최소 간격(봉수)
    max_span=140,                   # P1~P5 전체 길이 상한
    collinearity_tol=0.15,          # (1,3,5) / (2,4) 기울기 일관성 허용 비율(15%)
    overshoot_tol=0.002,            # P5의 1-3선 오버슈트 최소 비율(0.2%)
    reentry_buffer=0.001,           # 재진입/돌파 버퍼
    atr_period=14,
    rr_measure=1.0,                 # 타깃 = |EPA 교차점까지의 측정치| × 배수(간단화)
    stop_atr_mult=1.0,              # 손절 ATR 여유
    hold_bars=80                    # 타임아웃
):
    """
    반환 DataFrame:
      ['pattern','P1','P2','P3','P4','P5','EPA_y',
       'entry_long','exit_long','entry_short','exit_short',
       'stop_price','tp_price']
    pattern: 'WolfeBull'(강세), 'WolfeBear'(약세)
    P1..P5: 해당 봉의 가격(저가/고가) 저장
    EPA_y: 엔트리 시점의 EPA(1-4선) 값(참고)
    """
    df = df.copy()
    idx = df.index; n = len(df)
    Hflag, Lflag = find_pivots(df['High'], df['Low'], pivot_left, pivot_right)

    pivH = [(i, float(df['High'].iloc[i])) for i in range(n) if Hflag.iloc[i]]
    pivL = [(i, float(df['Low' ].iloc[i])) for i in range(n) if Lflag.iloc[i]]

    df['ATR'] = atr(df, period=atr_period)

    pattern = np.array([None]*n, dtype=object)
    P1s = np.full(n, np.nan); P2s = np.full(n, np.nan); P3s = np.full(n, np.nan)
    P4s = np.full(n, np.nan); P5s = np.full(n, np.nan)
    EPAy = np.full(n, np.nan)

    entry_long  = np.zeros(n, dtype=bool)
    exit_long   = np.zeros(n, dtype=bool)
    entry_short = np.zeros(n, dtype=bool)
    exit_short  = np.zeros(n, dtype=bool)
    stop_price  = np.full(n, np.nan)
    tp_price    = np.full(n, np.nan)

    start = max(0, n - lookback)

    # -------------------------
    # 강세 Wolfe (하락 채널에서 1-3-5 저점, 2-4 고점 / P5가 1-3선 아래로 오버슈트 후 반등)
    # -------------------------
    if len(pivL) >= 3 and len(pivH) >= 2:
        # 인덱스만 추출(시간순)
        Lbars = [b for b,_ in pivL]
        Hbars = [b for b,_ in pivH]
        for a in range(len(Lbars)-2):
            i1 = Lbars[a]
            if i1 < start:
                continue
            for b in range(a+1, len(Hbars)):
                i2 = Hbars[b]  # 2는 고점
                if i2 - i1 < min_sep: 
                    continue
                for c in range(b+1, len(Lbars)-1):
                    i3 = Lbars[c]
                    if i3 - i2 < min_sep: 
                        continue
                    for d in range(c+1, len(Hbars)):
                        i4 = Hbars[d]
                        if i4 - i3 < min_sep: 
                            continue
                        for e in range(d+1, len(Lbars)):
                            i5 = Lbars[e]
                            if i5 - i4 < min_sep: 
                                continue
                            if i5 - i1 > max_span:
                                continue
                            # 값
                            p1 = df['Low' ].iloc[i1]
                            p2 = df['High'].iloc[i2]
                            p3 = df['Low' ].iloc[i3]
                            p4 = df['High'].iloc[i4]
                            p5 = df['Low' ].iloc[i5]

                            # 채널 일관성: (1,3,5) 저점선과 (2,4) 고점선 기울기 부호 반대 & 절대값 유사
                            m13 = slope(i1, p1, i3, p3)
                            m35 = slope(i3, p3, i5, p5)
                            m24 = slope(i2, p2, i4, p4)
                            if not (m13 < 0 and m24 < 0):  # 하락 채널
                                continue
                            # (1,3)와 (3,5) 기울기 상대 오차
                            if abs(m35 - m13) / (abs(m13) + 1e-12) > collinearity_tol:
                                continue
                            # 1-3 직선 대비 P5 오버슈트(아래)
                            line_13_p5 = line_y(i1, p1, i3, p3, i5)
                            if not (p5 < line_13_p5 * (1 - overshoot_tol)):
                                continue

                            # 진입 트리거: P5 이후 가격이 2-4 상단선 돌파(종가 기준)
                            # 2-4 선 y값 계산용 함수
                            def y24(x): 
                                return line_y(i2, p2, i4, p4, x)
                            entered = False
                            for j in range(i5+1, min(i5+hold_bars, n-1)+1):
                                if df['Close'].iloc[j] > y24(j) * (1 + reentry_buffer):
                                    # 기록
                                    pattern[j] = 'WolfeBull'
                                    P1s[j], P2s[j], P3s[j], P4s[j], P5s[j] = p1, p2, p3, p4, p5
                                    EPAy[j] = line_y(i1, p1, i4, p4, j)  # EPA(1-4) y값
                                    entry_long[j] = True

                                    entry = df['Close'].iloc[j]
                                    atr_e = df['ATR'].iloc[j]
                                    # 손절: P5 아래 또는 2-4선 아래 - ATR 여유 중 더 보수적(더 낮은 값)
                                    sj = min(p5, y24(j)) - stop_atr_mult * atr_e
                                    # 타깃: EPA(1-4)와의 거리(보수적으로 현재가→EPA)
                                    target = abs(EPAy[j] - entry) * rr_measure
                                    tj = entry + target

                                    end_j = min(j + hold_bars, n-1)
                                    for k in range(j+1, end_j+1):
                                        if df['Low'].iloc[k] <= sj:
                                            exit_long[k] = True; stop_price[k] = sj; break
                                        if df['High'].iloc[k] >= tj:
                                            exit_long[k] = True;  tp_price[k]  = tj; break
                                    else:
                                        exit_long[end_j] = True
                                    entered = True
                                    break
                            if entered:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

    # -------------------------
    # 약세 Wolfe (상승 채널에서 1-3-5 고점, 2-4 저점 / P5가 1-3선 위로 오버슈트 후 하락)
    # -------------------------
    if len(pivH) >= 3 and len(pivL) >= 2:
        Hbars = [b for b,_ in pivH]
        Lbars = [b for b,_ in pivL]
        for a in range(len(Hbars)-2):
            i1 = Hbars[a]
            if i1 < start:
                continue
            for b in range(a+1, len(Lbars)):
                i2 = Lbars[b]
                if i2 - i1 < min_sep: 
                    continue
                for c in range(b+1, len(Hbars)-1):
                    i3 = Hbars[c]
                    if i3 - i2 < min_sep: 
                        continue
                    for d in range(c+1, len(Lbars)):
                        i4 = Lbars[d]
                        if i4 - i3 < min_sep: 
                            continue
                        for e in range(d+1, len(Hbars)):
                            i5 = Hbars[e]
                            if i5 - i4 < min_sep: 
                                continue
                            if i5 - i1 > max_span:
                                continue
                            p1 = df['High'].iloc[i1]
                            p2 = df['Low' ].iloc[i2]
                            p3 = df['High'].iloc[i3]
                            p4 = df['Low' ].iloc[i4]
                            p5 = df['High'].iloc[i5]

                            m13 = slope(i1, p1, i3, p3)
                            m35 = slope(i3, p3, i5, p5)
                            m24 = slope(i2, p2, i4, p4)
                            if not (m13 > 0 and m24 > 0):  # 상승 채널
                                continue
                            if abs(m35 - m13) / (abs(m13) + 1e-12) > collinearity_tol:
                                continue
                            line_13_p5 = line_y(i1, p1, i3, p3, i5)
                            if not (p5 > line_13_p5 * (1 + overshoot_tol)):  # 위로 오버슈트
                                continue

                            def y24(x):
                                return line_y(i2, p2, i4, p4, x)
                            entered = False
                            for j in range(i5+1, min(i5+hold_bars, n-1)+1):
                                if df['Close'].iloc[j] < y24(j) * (1 - reentry_buffer):
                                    pattern[j] = 'WolfeBear'
                                    P1s[j], P2s[j], P3s[j], P4s[j], P5s[j] = p1, p2, p3, p4, p5
                                    EPAy[j] = line_y(i1, p1, i4, p4, j)
                                    entry_short[j] = True

                                    entry = df['Close'].iloc[j]
                                    atr_e = df['ATR'].iloc[j]
                                    sj = max(p5, y24(j)) + stop_atr_mult * atr_e
                                    target = abs(entry - EPAy[j]) * rr_measure
                                    tj = entry - target

                                    end_j = min(j + hold_bars, n-1)
                                    for k in range(j+1, end_j+1):
                                        if df['High'].iloc[k] >= sj:
                                            exit_short[k] = True; stop_price[k] = sj; break
                                        if df['Low' ].iloc[k] <= tj:
                                            exit_short[k] = True;  tp_price[k]  = tj; break
                                    else:
                                        exit_short[end_j] = True
                                    entered = True
                                    break
                            if entered:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break

    out = pd.DataFrame({
        'pattern': pd.Series(pattern, index=idx, dtype='object'),
        'P1': P1s, 'P2': P2s, 'P3': P3s, 'P4': P4s, 'P5': P5s,
        'EPA_y': EPAy,
        'entry_long': entry_long, 'exit_long': exit_long,
        'entry_short': entry_short, 'exit_short': exit_short,
        'stop_price': stop_price, 'tp_price': tp_price
    }, index=idx)
    return out

# -----------------------------
# 예시
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')  # ['Open','High','Low','Close']
    # sig = detect_wolfe_wave(df,
    #     pivot_left=3, pivot_right=3,
    #     collinearity_tol=0.15, overshoot_tol=0.002)
    # print(sig.tail(120))
    pass
