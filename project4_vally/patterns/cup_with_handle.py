import numpy as np
import pandas as pd

# -----------------------------
# ATR
# -----------------------------
def atr(df: pd.DataFrame, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# 간단 Pivot (고점/저점)
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
# 손잡이 달린 컵 탐지 + 돌파 시그널
# -----------------------------
def detect_cup_with_handle(
    df: pd.DataFrame,
    # 컵 파라미터
    lookback=500,               # 최대 탐지 과거 구간
    min_cup_len=25,             # 컵 최소 길이(봉수)
    max_cup_len=140,            # 컵 최대 길이
    rim_diff_max=0.02,          # 좌/우 림 높이 상대차 허용치(2%)
    min_cup_depth=0.08,         # 컵 깊이 최소(좌림 대비 8% 이상 하락)
    smooth_ma=7,                # 컵 곡률 판단을 위한 단순 스무딩 길이
    # 손잡이 파라미터
    handle_max_len=25,          # 손잡이 최대 길이
    handle_depth_frac=0.35,     # 손잡이 깊이 ≤ 컵 깊이의 이 비율(예: 35%)
    handle_slope_abs_max=0.003, # 손잡이 채널 기울기 절댓값 상한(0.3%/bar)
    min_consolidation=4,        # 손잡이 최소 봉수
    # 엔트리/리스크
    buffer=0.001,               # 돌파 버퍼(0.1%)
    atr_period=14,
    rr_measure=1.0,             # 타깃=컵 깊이 × 배수
    stop_atr_mult=1.0,          # 손절 ATR 여유
    hold_bars=80,               # 타임아웃 청산
    pivot_left=3, pivot_right=3,# 피벗 민감도
    enable_inverse=False        # 역-컵핸들(약세)도 탐지할지
):
    """
    반환 DataFrame:
      ['pattern','cup_left','cup_right','cup_bottom','handle_high','handle_low',
       'entry_long','exit_long','stop_price','tp_price',
       'entry_short','exit_short']  # inverse 사용시
    pattern: 'CupHandle' 또는 'InvCupHandle'
    """
    df = df.copy()
    idx = df.index; n = len(df)

    # 스무딩으로 컵 곡률 노이즈 완화
    df['Close_s'] = df['Close'].rolling(smooth_ma, min_periods=1).mean()
    Hflag, Lflag = find_pivots(df['High'], df['Low'], pivot_left, pivot_right)
    pivH = [i for i in range(n) if Hflag.iloc[i]]
    pivL = [i for i in range(n) if Lflag.iloc[i]]

    df['ATR'] = atr(df, period=atr_period)

    pattern      = np.array([None]*n, dtype=object)
    cup_left     = np.full(n, np.nan)
    cup_right    = np.full(n, np.nan)
    cup_bottom   = np.full(n, np.nan)
    handle_high  = np.full(n, np.nan)
    handle_low   = np.full(n, np.nan)

    entry_long   = np.zeros(n, dtype=bool)
    exit_long    = np.zeros(n, dtype=bool)
    entry_short  = np.zeros(n, dtype=bool)
    exit_short   = np.zeros(n, dtype=bool)
    stop_price   = np.full(n, np.nan)
    tp_price     = np.full(n, np.nan)

    start = max(0, n - lookback)

    # -----------------------------
    # 보조: 구간에서 평행채널(손잡이) 적합
    # -----------------------------
    def fit_channel(seg_h, seg_l):
        m = len(seg_h)
        x = np.arange(m)
        up_s, up_b = np.polyfit(x, seg_h, 1)
        lo_s, lo_b = np.polyfit(x, seg_l, 1)
        up_line = up_s * x + up_b
        lo_line = lo_s * x + lo_b
        return up_s, lo_s, up_line, lo_line

    # =========== Cup with Handle (강세) ===========
    for iL in range(len(pivH)-1):
        left = pivH[iL]
        if left < start: 
            continue
        vL = df['High'].iloc[left]

        # 컵 오른쪽 림 후보 순회
        for iR in range(iL+1, len(pivH)):
            right = pivH[iR]
            cup_len = right - left
            if cup_len < min_cup_len or cup_len > max_cup_len:
                continue
            vR = df['High'].iloc[right]

            # 좌/우 림 높이 유사성
            if abs(vR - vL) / max(vL,1e-12) > rim_diff_max:
                continue

            # 컵 바닥: left~right 구간의 최저 저가
            mid_slice = slice(left, right+1)
            bottom_bar = df['Low'].iloc[mid_slice].idxmin()
            # idxmin은 라벨 반환 → 포지션으로 변환
            bottom_i = df.index.get_loc(bottom_bar)
            vB = df['Low'].iloc[bottom_i]

            # 컵 깊이
            cup_depth = (vL - vB) / max(vL, 1e-12)
            if cup_depth < min_cup_depth:
                continue

            # 컵 곡률(스무딩된 종가가 바닥 근방에서 U자형): 바닥이 중앙 30~70% 구간 내에 있으면 가중
            center_ok = (0.3*cup_len <= (bottom_i-left) <= 0.7*cup_len)

            # 손잡이 구간: right 이후 최대 handle_max_len 내
            h_start = right + 1
            h_end   = min(right + handle_max_len, n-1)
            if h_start + min_consolidation > h_end:
                continue

            seg_h = df['High'].iloc[h_start:h_end+1].values
            seg_l = df['Low' ].iloc[h_start:h_end+1].values
            if len(seg_h) < min_consolidation:
                continue

            up_s, lo_s, up_line, lo_line = fit_channel(seg_h, seg_l)

            # 손잡이 채널 기울기 제한(완만한 하향/횡보 조정)
            if not (abs(up_s) <= handle_slope_abs_max and abs(lo_s) <= handle_slope_abs_max):
                continue

            # 손잡이 깊이 제한(컵 깊이의 일부 이내)
            h_low = seg_l.min()
            h_high = seg_h.max()
            handle_depth = (vR - h_low) / max(vR,1e-12)
            if handle_depth > handle_depth_frac * cup_depth:
                continue

            # 손잡이 최소 체류/압축(간단판): 종가가 채널 내부에 충분 비율 존재
            seg_c = df['Close'].iloc[h_start:h_end+1].values
            inside = (seg_c <= up_line*(1+0.003)) & (seg_c >= lo_line*(1-0.003))
            if inside.mean() < 0.6:
                continue

            # 손잡이 고점/저점
            handle_hi = h_high
            handle_lo = h_low

            # 돌파 트리거: 손잡이 구간 이후 첫 번째로 종가가 handle_hi를 상향 돌파
            broke = False
            for j in range(h_start+min_consolidation, h_end+1):
                if df['Close'].iloc[j] > handle_hi * (1 + buffer):
                    # 기록
                    cup_left[j]   = vL
                    cup_right[j]  = vR
                    cup_bottom[j] = vB
                    handle_high[j]= handle_hi
                    handle_low[j] = handle_lo
                    pattern[j]    = 'CupHandle'
                    entry_long[j] = True

                    # 리스크 관리
                    entry = df['Close'].iloc[j]
                    atr_e = df['ATR'].iloc[j]
                    # 손절: 손잡이 저점 아래 또는 (넥라인≈우림) 아래 중 더 보수적
                    sj = min(handle_lo, vR) - stop_atr_mult * atr_e
                    # 타깃: 컵 깊이 × rr_measure
                    target = (vR - vB) * rr_measure
                    tj = entry + target

                    end_j = min(j + hold_bars, n-1)
                    for k in range(j+1, end_j+1):
                        # 동시충돌 보수적으로 손절 우선
                        if df['Low'].iloc[k] <= sj:
                            exit_long[k] = True; stop_price[k] = sj; break
                        if df['High'].iloc[k] >= tj:
                            exit_long[k] = True;  tp_price[k]  = tj; break
                    else:
                        exit_long[end_j] = True
                    broke = True
                    break
            if broke:
                break  # 한 컵당 한 번만 엔트리

    # =========== Inverse Cup with Handle (선택, 약세) ===========
    if enable_inverse:
        for iL in range(len(pivL)-1):
            left = pivL[iL]
            if left < start:
                continue
            vL = df['Low'].iloc[left]
            for iR in range(iL+1, len(pivL)):
                right = pivL[iR]
                cup_len = right - left
                if cup_len < min_cup_len or cup_len > max_cup_len:
                    continue
                vR = df['Low'].iloc[right]
                if abs(vR - vL) / max(vL,1e-12) > rim_diff_max:
                    continue
                mid_slice = slice(left, right+1)
                top_bar = df['High'].iloc[mid_slice].idxmax()
                top_i = df.index.get_loc(top_bar)
                vT = df['High'].iloc[top_i]
                cup_depth = (vT - vL) / max(vT,1e-12)
                if cup_depth < min_cup_depth:
                    continue
                h_start = right + 1
                h_end   = min(right + handle_max_len, n-1)
                if h_start + min_consolidation > h_end:
                    continue
                seg_h = df['High'].iloc[h_start:h_end+1].values
                seg_l = df['Low' ].iloc[h_start:h_end+1].values
                up_s, lo_s, up_line, lo_line = fit_channel(seg_h, seg_l)
                if not (abs(up_s) <= handle_slope_abs_max and abs(lo_s) <= handle_slope_abs_max):
                    continue
                h_high = seg_h.max(); h_low = seg_l.min()
                handle_depth = (h_high - vR) / max(vR,1e-12)
                if handle_depth > handle_depth_frac * cup_depth:
                    continue
                seg_c = df['Close'].iloc[h_start:h_end+1].values
                inside = (seg_c <= up_line*(1+0.003)) & (seg_c >= lo_line*(1-0.003))
                if inside.mean() < 0.6:
                    continue
                handle_hi = h_high; handle_lo = h_low

                for j in range(h_start+min_consolidation, h_end+1):
                    if df['Close'].iloc[j] < handle_lo * (1 - buffer):
                        cup_left[j]   = vL
                        cup_right[j]  = vR
                        cup_bottom[j] = vT  # 역패턴에선 ‘천장’
                        handle_high[j]= handle_hi
                        handle_low[j] = handle_lo
                        pattern[j]    = 'InvCupHandle'
                        entry_short[j]= True

                        entry = df['Close'].iloc[j]
                        atr_e = df['ATR'].iloc[j]
                        sj = max(handle_hi, vR) + stop_atr_mult * atr_e
                        target = (vT - vR) * rr_measure
                        tj = entry - target

                        end_j = min(j + hold_bars, n-1)
                        for k in range(j+1, end_j+1):
                            if df['High'].iloc[k] >= sj:
                                exit_short[k] = True; stop_price[k] = sj; break
                            if df['Low'].iloc[k]  <= tj:
                                exit_short[k]  = True; tp_price[k]  = tj; break
                        else:
                            exit_short[end_j] = True
                        break

    out = pd.DataFrame({
        'pattern': pd.Series(pattern, index=idx, dtype='object'),
        'cup_left': cup_left, 'cup_right': cup_right, 'cup_bottom': cup_bottom,
        'handle_high': handle_high, 'handle_low': handle_low,
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
    # sig = detect_cup_with_handle(df,
    #     min_cup_len=25, max_cup_len=140, rim_diff_max=0.02, min_cup_depth=0.08,
    #     handle_max_len=25, handle_depth_frac=0.35)
    # print(sig.tail(80))
    pass
