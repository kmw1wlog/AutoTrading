import numpy as np
import pandas as pd

# -----------------------------
# ATR (리스크/목표 계산용)
# -----------------------------
def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# 깃발형(Flag) 탐지 + 돌파 시그널
# -----------------------------
def detect_flag_breakout(
    df: pd.DataFrame,
    pole_len=12,            # 폴(충격파) 길이(봉수)
    pole_ret=0.06,          # 폴 최소 수익률(절댓값) 예: 6% 이상
    window=30,              # 깃발(조정) 검사 구간 길이
    slope_abs_max=0.002,    # 깃발 채널 기울기 절댓값 상한(완만해야 함; 0.2%/bar)
    span_vs_pole_max=0.5,   # 깃발 높이 / 폴 높이 최대 비율
    inside_ratio_min=0.75,  # 깃발 기간 동안 가격이 채널 안에 머문 비율
    touch_tol=0.003,        # 채널 터치 허용 오차(0.3%)
    min_touches=3,          # 상단/하단 최소 터치 수
    buffer=0.001,           # 돌파 판정 버퍼(0.1%)
    atr_period=14,
    rr_target=2.0,          # R:R 목표
    hold_bars=60            # 타임아웃 청산
):
    """
    반환: DataFrame
      columns = [
        'upper','lower','flag_type','in_flag',
        'long_entry','short_entry','long_exit','short_exit',
        'stop_price','tp_price'
      ]
    flag_type: 'bull_flag'(상승 깃발), 'bear_flag'(하락 깃발), np.nan
    입력 df: OHLC 컬럼 필요
    """
    df = df.copy()
    idx = df.index
    n = len(df)

    # 선계산
    df['ATR'] = atr(df, period=atr_period)
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values

    # 결과 버퍼
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    in_flag = np.zeros(n, dtype=bool)
    flag_type = np.array([None]*n, dtype=object)
    long_entry = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)

    # 도우미: 특정 구간에 대해 평행 채널을 회귀로 적합
    def fit_channel(seg_h, seg_l):
        m = len(seg_h)
        x = np.arange(m)
        # 상단선: 고가 회귀, 하단선: 저가 회귀
        up_slope, up_intercept = np.polyfit(x, seg_h, 1)
        lo_slope, lo_intercept = np.polyfit(x, seg_l, 1)
        up_line = up_slope * x + up_intercept
        lo_line = lo_slope * x + lo_intercept
        return up_slope, lo_slope, up_line, lo_line

    # 메인 루프: i 시점에서 "직전 pole_len + window" 구조를 검사
    start_i = pole_len + window
    for i in range(start_i, n-1):  # i+1에서 돌파 체크할 것이므로 n-1까지
        pole_start = i - window - pole_len + 1
        pole_end   = i - window
        flag_start = i - window + 1
        flag_end   = i

        # 구간 배열 슬라이스
        c_pole_s = close[pole_start]
        c_pole_e = close[pole_end]
        pole_rtn = (c_pole_e / c_pole_s) - 1.0
        pole_height = abs(c_pole_e - c_pole_s)

        seg_h = high[flag_start:flag_end+1]
        seg_l = low [flag_start:flag_end+1]
        seg_c = close[flag_start:flag_end+1]
        m = len(seg_c)
        x = np.arange(m)

        # 폴 조건: 절댓값 수익률이 pole_ret 이상
        if abs(pole_rtn) < pole_ret or pole_height == 0:
            continue

        # 채널 적합
        up_s, lo_s, up_line, lo_line = fit_channel(seg_h, seg_l)

        # 기울기 제약(깃발은 완만한 역추세 평행 채널)
        if not (abs(up_s) <= slope_abs_max and abs(lo_s) <= slope_abs_max):
            continue

        # 방향성 제약
        #   상승 깃발: 이전 폴이 + 이고, 깃발 기울기는 역추세(대체로 하락 또는 옆) → up_s, lo_s <= +slope_abs_max (이미 절댓값 제한)
        #   하락 깃발: 이전 폴이 - 이고, 깃발 기울기는 역추세(대체로 상승 또는 옆)
        is_bull_candidate = pole_rtn > 0 and (up_s <= slope_abs_max) and (lo_s <= slope_abs_max)
        is_bear_candidate = pole_rtn < 0 and (up_s >= -slope_abs_max) and (lo_s >= -slope_abs_max)

        if not (is_bull_candidate or is_bear_candidate):
            continue

        # 깃발 높이(채널 폭) 제한: 폴 높이 대비 과도하면 탈락
        span_now = (up_line[-1] - lo_line[-1])
        if span_now <= 0 or span_now > pole_height * span_vs_pole_max:
            continue

        # 내부 체류 비율
        inside_mask = (seg_c <= up_line*(1+touch_tol)) & (seg_c >= lo_line*(1-touch_tol))
        if inside_mask.mean() < inside_ratio_min:
            continue

        # 터치 수(상단/하단에 충분히 닿았는지)
        up_touches = np.sum(np.abs(seg_h - up_line)/np.maximum(1e-12, up_line) <= touch_tol)
        lo_touches = np.sum(np.abs(seg_l - lo_line)/np.maximum(1e-12, lo_line) <= touch_tol)
        if up_touches < min_touches or lo_touches < min_touches:
            continue

        # 패턴 인정
        upper[i] = up_line[-1]
        lower[i] = lo_line[-1]
        in_flag[i] = True
        if is_bull_candidate:
            flag_type[i] = 'bull_flag'
        elif is_bear_candidate:
            flag_type[i] = 'bear_flag'

        # 돌파 판정: 다음 봉(i+1) 종가 기준
        nxt = i + 1
        if flag_type[i] == 'bull_flag':
            if close[nxt] > upper[i] * (1 + buffer):
                long_entry[nxt] = True
        elif flag_type[i] == 'bear_flag':
            if close[nxt] < lower[i] * (1 - buffer):
                short_entry[nxt] = True

    # ===== 진입 후 청산 로직(ATR 기반 R:R + 타임아웃) =====
    long_exit  = np.zeros(n, dtype=bool)
    short_exit = np.zeros(n, dtype=bool)
    stop_price = np.full(n, np.nan)
    tp_price   = np.full(n, np.nan)

    def walkout(entry_mask, is_long=True):
        entries = np.where(entry_mask)[0]
        for ei in entries:
            end_i = min(ei + hold_bars, n-1)
            entry = close[ei]
            atr_e = df['ATR'].iloc[ei]
            if np.isnan(atr_e) or atr_e == 0:
                continue

            if is_long:
                # 보수적 손절: 엔트리 시점 아래쪽(하단선/ATR 중 더 타이트한 쪽)
                sj_line = lower[ei] if not np.isnan(lower[ei]) else entry - atr_e
                sj = min(entry - atr_e, sj_line)
                tj = entry + rr_target * atr_e
                for j in range(ei+1, end_i+1):
                    if low[j] <= sj:
                        long_exit[j] = True; stop_price[j] = sj; break
                    if high[j] >= tj:
                        long_exit[j] = True; tp_price[j] = tj;  break
                else:
                    long_exit[end_i] = True
            else:
                sj_line = upper[ei] if not np.isnan(upper[ei]) else entry + atr_e
                sj = max(entry + atr_e, sj_line)
                tj = entry - rr_target * atr_e
                for j in range(ei+1, end_i+1):
                    if high[j] >= sj:
                        short_exit[j] = True; stop_price[j] = sj; break
                    if low[j]  <= tj:
                        short_exit[j] = True; tp_price[j] = tj;  break
                else:
                    short_exit[end_i] = True

    walkout(long_entry,  True)
    walkout(short_entry, False)

    out = pd.DataFrame({
        'upper': upper, 'lower': lower,
        'flag_type': pd.Series(flag_type, index=idx, dtype="object"),
        'in_flag': in_flag,
        'long_entry': long_entry, 'short_entry': short_entry,
        'long_exit': long_exit, 'short_exit': short_exit,
        'stop_price': stop_price, 'tp_price': tp_price
    }, index=idx)
    return out

# -----------------------------
# 예시
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')  # ['Open','High','Low','Close']
    # sig = detect_flag_breakout(df)
    # print(sig.tail(50))
    pass
