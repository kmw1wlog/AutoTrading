import numpy as np
import pandas as pd

# --------------------------------------------
# 유틸: Pivot 고점/저점 탐지 (왼쪽 nL, 오른쪽 nR 바보다 높은/낮은 지점)
# --------------------------------------------
def find_pivots(high, low, nL=3, nR=3):
    H = pd.Series(False, index=high.index)
    L = pd.Series(False, index=low.index)

    for i in range(nL, len(high)-nR):
        # 고점 피벗: 좌우보다 높음
        if high.iloc[i] == high.iloc[i-nL:i+nR+1].max():
            H.iloc[i] = True
        # 저점 피벗: 좌우보다 낮음
        if low.iloc[i] == low.iloc[i-nL:i+nR+1].min():
            L.iloc[i] = True
    return H, L

# --------------------------------------------
# 유틸: 최근 K개의 피벗으로 선형회귀(상단/하단 추세선) 적합
# 반환: slope, intercept, fitted_line(series)
# --------------------------------------------
def fit_trendline(series, idx, K=5):
    # series: 고점 또는 저점 값 (피벗만 NaN 아닌 값)
    pts = series.dropna().tail(K)
    if len(pts) < 2:
        return np.nan, np.nan, pd.Series(np.nan, index=idx)

    # x를 0..N-1로 매핑(수치 안정)
    x = np.arange(len(series))
    sel = np.isin(series.index, pts.index)
    x_sel = x[sel]
    y_sel = series.values[sel]

    # 1차 회귀
    slope, intercept = np.polyfit(x_sel, y_sel, 1)

    fitted = pd.Series(slope * x + intercept, index=idx)
    return slope, intercept, fitted

# --------------------------------------------
# ATR(손절폭, 트레일링 등에 사용)
# --------------------------------------------
def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# --------------------------------------------
# 삼각수렴 패턴 탐지 + 돌파 시그널
# --------------------------------------------
def detect_triangle_breakout(
    df: pd.DataFrame,
    pivot_left=3, pivot_right=3,     # 피벗 민감도
    K=5,                              # 추세선 적합에 사용할 피벗 개수
    window=120,                       # 수렴 검증 구간(최근 N봉)
    contraction_ratio=0.6,            # 초반 폭 대비 말기 폭 축소 비율 임계 (<= 이하면 수렴으로 간주)
    buffer=0.001,                     # 돌파 판정 버퍼(예: 0.1% = 0.001)
    atr_period=14,
    rr_target=2.0,                    # R:R 목표(익절)
    hold_bars=60                      # 타임아웃 청산
):
    """
    반환:
      signals: DataFrame [upper, lower, in_triangle, long_entry, short_entry, long_exit, short_exit, stop_price, tp_price]
    사용 가정:
      df: ['Open','High','Low','Close'] 포함, DatetimeIndex
    """
    df = df.copy()
    Hflag, Lflag = find_pivots(df['High'], df['Low'], nL=pivot_left, nR=pivot_right)

    # 피벗 값 시리즈(나머지 NaN)
    Hvals = pd.Series(np.nan, index=df.index)
    Lvals = pd.Series(np.nan, index=df.index)
    Hvals[Hflag] = df['High'][Hflag]
    Lvals[Lflag] = df['Low'][Lflag]

    # 최근 window 범위 내에서만 선 적합(이동 윈도우 방식)
    upper = pd.Series(np.nan, index=df.index)
    lower = pd.Series(np.nan, index=df.index)
    upslope = pd.Series(np.nan, index=df.index)
    lowslope = pd.Series(np.nan, index=df.index)

    for i in range(window, len(df)):
        seg_idx = df.index[i-window:i+1]
        # 세그먼트 피벗만 남기기
        Hseg = Hvals.loc[seg_idx]
        Lseg = Lvals.loc[seg_idx]

        uslope, uinter, uf = fit_trendline(Hseg, seg_idx, K=K)
        lslope, linter, lf = fit_trendline(Lseg, seg_idx, K=K)

        upper.iloc[i-window:i+1] = uf.values
        lower.iloc[i-window:i+1] = lf.values
        upslope.iloc[i] = uslope
        lowslope.iloc[i] = lslope

    # 수렴(상단 하락, 하단 상승) + 폭 축소 확인
    span = (upper - lower)
    # 초기 폭과 말기 폭 비교
    span_initial = span.shift(window-1)  # 윈도우 시작 시점의 폭
    span_final = span                    # 현재 시점 폭

    # 조건: 상단 기울기 <= 0, 하단 기울기 >= 0, 말기 폭 <= 초기 폭 * contraction_ratio
    converging = (upslope <= 0) & (lowslope >= 0) & (span_final <= span_initial * contraction_ratio)

    # 최근 몇 봉 동안 종가가 upper/lower 사이에 있었는가(삼각 안에 머무름)
    inside = (df['Close'] <= upper*(1+buffer)) & (df['Close'] >= lower*(1-buffer))
    in_triangle = converging & inside

    # 돌파 판정(직전에는 삼각 내, 현재 봉에서 상·하단 돌파)
    prev_inside = in_triangle.shift(1).fillna(False)
    break_up = prev_inside & (df['Close'] > upper*(1+buffer))
    break_dn = prev_inside & (df['Close'] < lower*(1-buffer))

    # 진입 시그널(시장가 가정: 돌파 봉 종가 또는 다음 봉 시가 진입은 전략에 따라 조정)
    long_entry = break_up
    short_entry = break_dn

    # 리스크 관리(ATR 기반 손절/익절)
    df['ATR'] = atr(df, period=atr_period)
    risk_long = df['ATR']  # 1R
    risk_short = df['ATR']

    # 손절: 돌파 반대선 혹은 ATR 기반 더 타이트한 쪽 선택(보수적)
    stop_long_line = lower
    stop_short_line = upper

    stop_long = np.minimum(df['Close'] - risk_long, stop_long_line)  # 보수적: 더 낮은 값
    stop_short = np.maximum(df['Close'] + risk_short, stop_short_line)

    # 목표가(측정치 방식 대신 간단히 RR로): 엔트리 ± rr_target*ATR
    tp_long = df['Close'] + rr_target * df['ATR']
    tp_short = df['Close'] - rr_target * df['ATR']

    # 체결 이후 관리(단일 포지션 가정, 시그널-기준 벡터화)
    # 상태머신 없이도 근사 구현: 엔트리 후 hold_bars 내에 TP/SL/타임아웃 중 먼저 발생
    # 결과 신호를 위한 열 초기화
    long_exit = pd.Series(False, index=df.index)
    short_exit = pd.Series(False, index=df.index)
    stop_price = pd.Series(np.nan, index=df.index)
    tp_price = pd.Series(np.nan, index=df.index)

    def process_side(entry_mask, is_long=True):
        # 엔트리 인덱스
        entries = list(np.where(entry_mask)[0])
        exits_local = pd.Series(False, index=df.index)
        stop_p = pd.Series(np.nan, index=df.index)
        tp_p = pd.Series(np.nan, index=df.index)

        for ei in entries:
            end_i = min(ei + hold_bars, len(df)-1)

            # 각 바에서 선/목표/손절 라인 업데이트 값을 사용
            # 익절/손절 발생 여부 체크
            triggered = False
            for j in range(ei+1, end_i+1):
                if is_long:
                    # 고가가 목표가 이상 먼저 닿거나, 저가가 손절가 이하 먼저 닿는지 순서 판단
                    tpj = tp_long.iloc[ei]  # 엔트리 시점 ATR 기준 고정
                    sj = stop_long.iloc[ei]
                    # 우선순위: 동시충돌 시 보수적으로 손절 우선 처리 가능 (시장 현실 반영)
                    if df['Low'].iloc[j] <= sj:
                        exits_local.iloc[j] = True
                        stop_p.iloc[j] = sj
                        triggered = True
                        break
                    if df['High'].iloc[j] >= tpj:
                        exits_local.iloc[j] = True
                        tp_p.iloc[j] = tpj
                        triggered = True
                        break
                else:
                    tpj = tp_short.iloc[ei]
                    sj = stop_short.iloc[ei]
                    if df['High'].iloc[j] >= sj:
                        exits_local.iloc[j] = True
                        stop_p.iloc[j] = sj
                        triggered = True
                        break
                    if df['Low'].iloc[j] <= tpj:
                        exits_local.iloc[j] = True
                        tp_p.iloc[j] = tpj
                        triggered = True
                        break
            # 타임아웃
            if not triggered:
                exits_local.iloc[end_i] = True

        return exits_local, stop_p, tp_p

    # 롱/숏 각각 처리
    l_exit, l_sp, l_tp = process_side(long_entry, is_long=True)
    s_exit, s_sp, s_tp = process_side(short_entry, is_long=False)

    long_exit |= l_exit
    short_exit |= s_exit
    stop_price.update(l_sp.fillna(method='ffill'))
    stop_price.update(s_sp.fillna(method='ffill'))
    tp_price.update(l_tp.fillna(method='ffill'))
    tp_price.update(s_tp.fillna(method='ffill'))

    signals = pd.DataFrame({
        'upper': upper,
        'lower': lower,
        'in_triangle': in_triangle,
        'long_entry': long_entry,
        'short_entry': short_entry,
        'long_exit': long_exit,
        'short_exit': short_exit,
        'stop_price': stop_price,
        'tp_price': tp_price
    }, index=df.index)

    return signals

# --------------------------------------------
# 예시 사용법
# --------------------------------------------
if __name__ == "__main__":
    # df는 DatetimeIndex, 컬럼 ['Open','High','Low','Close']
    # 예: csv 불러오기
    # df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')
    # signals = detect_triangle_breakout(df)
    # print(signals.tail(20))
    pass
