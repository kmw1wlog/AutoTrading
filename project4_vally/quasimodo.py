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
        if low.iloc[i]  == low.iloc[i-nL:i+nR+1].min():
            L.iloc[i] = True
    return H, L

# -----------------------------
# ATR
# -----------------------------
def atr(df: pd.DataFrame, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# 콰지모도(QM) 탐지 + 엔트리/청산
# -----------------------------
def detect_quasimodo(
    df: pd.DataFrame,
    pivot_left=3, pivot_right=3,   # 피벗 민감도
    lookback=500,                  # 스캔 최대 과거 구간
    hl_min_raise=0.005,            # (약세 QM) L2 > L1 최소 상승률(상승 추세 확인)
    hh_min_gain=0.005,             # H2 > H1 최소 상승률(머리 상방 돌파)
    bos_break=0.002,               # BOS 강도: (약세) L3 < L2*(1-bos_break)
    rs_tolerance=0.004,            # 우견 존: 가격이 LS(H1) 근처 ±0.4% 이내
    max_span=120,                  # 좌견~우견 전체 길이 상한(봉수)
    buffer=0.001,                  # 돌파/재이탈 버퍼
    atr_period=14,
    rr_measure=1.0,                # 타깃 = |Head - BOS레벨| × 배수
    stop_atr_mult=1.0,             # 손절 ATR 여유
    hold_bars=80,                  # 타임아웃 청산
    enable_bear=True,              # 약세 QM(숏)
    enable_bull=True               # 강세 QM(롱)
):
    """
    반환 DataFrame:
      ['pattern','LS','Head','RS_level','BOS_level',
       'entry_short','exit_short','entry_long','exit_long',
       'stop_price','tp_price']
    pattern: 'QM_Bear' 또는 'QM_Bull'
    정의(약세):
      LS=H1(좌견고점), Head=H2(더 높은 고점), BOS_level=L2(깨진 저점),
      RS_level≈H1(좌견고점 부근 재테스트 후 하락)
    강세는 반대 구조.
    """
    df = df.copy()
    idx = df.index; n = len(df)
    df['ATR'] = atr(df, period=atr_period)

    # 피벗 나열(시간순)
    Hflag, Lflag = find_pivots(df['High'], df['Low'], nL=pivot_left, nR=pivot_right)
    pivH = [(i, float(df['High'].iloc[i])) for i in range(n) if Hflag.iloc[i]]
    pivL = [(i, float(df['Low' ].iloc[i])) for i in range(n) if Lflag.iloc[i]]

    # 결과 버퍼
    pattern     = np.array([None]*n, dtype=object)
    LS          = np.full(n, np.nan)
    Head        = np.full(n, np.nan)
    RS_level    = np.full(n, np.nan)
    BOS_level   = np.full(n, np.nan)
    entry_short = np.zeros(n, dtype=bool)
    exit_short  = np.zeros(n, dtype=bool)
    entry_long  = np.zeros(n, dtype=bool)
    exit_long   = np.zeros(n, dtype=bool)
    stop_price  = np.full(n, np.nan)
    tp_price    = np.full(n, np.nan)

    start = max(0, n - lookback)

    # ----------------- 약세 QM (Uptrend → BOS↓ → RS(공급) → 하락) -----------------
    if enable_bear:
        # 필요한 피벗 시퀀스: H1(LS) < L1 < H2(Head) < L2(BOS기준 저점) < L3(LL: BOS 확인) < HR(RS 터치)
        for a in range(len(pivH)-1):
            h1_i, h1_v = pivH[a]
            if h1_i < start: 
                continue
            # H1 이후의 L1
            L1_cands = [p for p in pivL if p[0] > h1_i]
            if not L1_cands: 
                continue
            l1_i, l1_v = L1_cands[0]

            # L1 이후의 H2(Head)
            H2_cands = [p for p in pivH if p[0] > l1_i]
            if not H2_cands: 
                continue
            h2_i, h2_v = H2_cands[0]
            if (h2_v - h1_v)/max(h1_v,1e-12) < hh_min_gain:
                continue  # 머리 상방 돌파 약함

            # H2 이후의 L2(상승추세의 HL)
            L2_cands = [p for p in pivL if p[0] > h2_i]
            if not L2_cands: 
                continue
            l2_i, l2_v = L2_cands[0]
            if (l2_v - l1_v)/max(l1_v,1e-12) < hl_min_raise:
                continue  # HL 상승 추세가 약함

            # BOS: L3가 L2를 하향 이탈
            L3_cands = [p for p in pivL if p[0] > l2_i]
            if not L3_cands: 
                continue
            l3_i, l3_v = L3_cands[0]
            if l3_v >= l2_v * (1 - bos_break):
                continue  # 하방 구조 이탈 미확정

            # 우견(RS): LS(H1) 가격대 ±tol 재접근(리테스트)
            RS_cands = [p for p in pivH if p[0] > l3_i]
            if not RS_cands: 
                continue
            # RS 존: |RS - H1| / H1 <= rs_tolerance
            RS_pick = None
            for hr_i, hr_v in RS_cands:
                if abs(hr_v - h1_v)/max(h1_v,1e-12) <= rs_tolerance:
                    RS_pick = (hr_i, hr_v)
                    break
            if RS_pick is None:
                continue
            hr_i, hr_v = RS_pick

            # 전체 길이 제한
            if hr_i - h1_i > max_span:
                continue

            # 엔트리 트리거: RS 터치 이후, 종가가 RS 레벨 아래로 재이탈(약간의 버퍼)
            for j in range(hr_i+1, min(hr_i+hold_bars, n-1)+1):
                if df['Close'].iloc[j] < hr_v * (1 - buffer):
                    # 기록 (j 시점 엔트리)
                    pattern[j]   = 'QM_Bear'
                    LS[j]        = h1_v
                    Head[j]      = h2_v
                    RS_level[j]  = hr_v
                    BOS_level[j] = l2_v
                    entry_short[j] = True

                    entry  = df['Close'].iloc[j]
                    atr_e  = df['ATR'].iloc[j]
                    # 손절: RS 위 또는 Head 아래 중 더 보수적(더 높은 값)
                    sj = max(hr_v, h2_v) + stop_atr_mult * atr_e
                    # 타깃: |Head - BOS| × 배수
                    target = abs(h2_v - l2_v) * rr_measure
                    tj = entry - target

                    end_j = min(j + hold_bars, n-1)
                    for k in range(j+1, end_j+1):
                        if df['High'].iloc[k] >= sj:
                            exit_short[k] = True; stop_price[k] = sj; break
                        if df['Low'].iloc[k]  <= tj:
                            exit_short[k] = True;  tp_price[k]  = tj; break
                    else:
                        exit_short[end_j] = True
                    break  # 한 패턴 1회 엔트리

    # ----------------- 강세 QM (Downtrend → BOS↑ → RS(수요) → 상승) -----------------
    if enable_bull:
        # 필요한 시퀀스: L1(LS) < H1 < L2(Head: 더 낮은 저점) < H2(BOS기준 고점) < H3(HH: BOS 확인) < LR(RS 터치)
        for a in range(len(pivL)-1):
            l1_i, l1_v = pivL[a]
            if l1_i < start:
                continue
            H1_cands = [p for p in pivH if p[0] > l1_i]
            if not H1_cands:
                continue
            h1_i, h1_v = H1_cands[0]

            L2_cands = [p for p in pivL if p[0] > h1_i]
            if not L2_cands:
                continue
            l2_i, l2_v = L2_cands[0]
            # Head가 더 낮아야 함(하락 지속)
            if (l1_v - l2_v)/max(l1_v,1e-12) < hh_min_gain:
                continue

            H2_cands = [p for p in pivH if p[0] > l2_i]
            if not H2_cands:
                continue
            h2_i, h2_v = H2_cands[0]
            # 하락 추세에서의 LH→BOS 후보(상방)
            if (h2_v - h1_v)/max(h1_v,1e-12) < hl_min_raise:
                continue

            H3_cands = [p for p in pivH if p[0] > h2_i]
            if not H3_cands:
                continue
            h3_i, h3_v = H3_cands[0]
            if h3_v <= h2_v * (1 + bos_break):
                continue  # 상방 BOS 미확정

            # RS: 좌견 저점(L1) 레벨로 리테스트
            RS_cands = [p for p in pivL if p[0] > h3_i]
            if not RS_cands:
                continue
            LR_pick = None
            for lr_i, lr_v in RS_cands:
                if abs(lr_v - l1_v)/max(l1_v,1e-12) <= rs_tolerance:
                    LR_pick = (lr_i, lr_v)
                    break
            if LR_pick is None:
                continue
            lr_i, lr_v = LR_pick

            if lr_i - l1_i > max_span:
                continue

            # 엔트리: RS 터치 이후, 종가가 RS 레벨 위로 재이탈
            for j in range(lr_i+1, min(lr_i+hold_bars, n-1)+1):
                if df['Close'].iloc[j] > lr_v * (1 + buffer):
                    pattern[j]    = 'QM_Bull'
                    LS[j]         = l1_v
                    Head[j]       = l2_v
                    RS_level[j]   = lr_v
                    BOS_level[j]  = h2_v
                    entry_long[j] = True

                    entry = df['Close'].iloc[j]
                    atr_e = df['ATR'].iloc[j]
                    # 손절: RS 아래 또는 Head 위 중 더 보수적(더 낮은 값)
                    sj = min(lr_v, l2_v) - stop_atr_mult * atr_e
                    target = abs(h2_v - l2_v) * rr_measure
                    tj = entry + target

                    end_j = min(j + hold_bars, n-1)
                    for k in range(j+1, end_j+1):
                        if df['Low'].iloc[k] <= sj:
                            exit_long[k] = True; stop_price[k] = sj; break
                        if df['High'].iloc[k] >= tj:
                            exit_long[k] = True;  tp_price[k]  = tj; break
                    else:
                        exit_long[end_j] = True
                    break

    out = pd.DataFrame({
        'pattern': pd.Series(pattern, index=idx, dtype='object'),
        'LS': LS, 'Head': Head, 'RS_level': RS_level, 'BOS_level': BOS_level,
        'entry_short': entry_short, 'exit_short': exit_short,
        'entry_long':  entry_long,  'exit_long':  exit_long,
        'stop_price':  stop_price,  'tp_price':   tp_price
    }, index=idx)
    return out

# -----------------------------
# 예시
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')
    # sig = detect_quasimodo(df,
    #     pivot_left=3, pivot_right=3,
    #     hl_min_raise=0.005, hh_min_gain=0.005,
    #     bos_break=0.002, rs_tolerance=0.004)
    # print(sig.tail(100))
    pass
