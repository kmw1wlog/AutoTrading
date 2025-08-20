import numpy as np
import pandas as pd

# -----------------------------
# Pivot 고점/저점 탐지
# -----------------------------
def find_pivots(high, low, nL=3, nR=3):
    H = pd.Series(False, index=high.index)
    L = pd.Series(False, index=low.index)
    for i in range(nL, len(high)-nR):
        if high.iloc[i] == high.iloc[i-nL:i+nR+1].max():
            H.iloc[i] = True
        if low.iloc[i] == low.iloc[i-nL:i+nR+1].min():
            L.iloc[i] = True
    return H, L

# -----------------------------
# 선형 회귀 추세선 적합 (피벗만 사용)
# -----------------------------
def fit_trendline(series_with_pivots, idx, K=5):
    pts = series_with_pivots.dropna().tail(K)
    if len(pts) < 2:
        return np.nan, np.nan, pd.Series(np.nan, index=idx)
    x = np.arange(len(series_with_pivots))
    sel = np.isin(series_with_pivots.index, pts.index)
    x_sel = x[sel]
    y_sel = series_with_pivots.values[sel]
    slope, intercept = np.polyfit(x_sel, y_sel, 1)
    fitted = pd.Series(slope * x + intercept, index=idx)
    return slope, intercept, fitted

# -----------------------------
# ATR
# -----------------------------
def atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([(h-l), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# -----------------------------
# 쐐기형 탐지 + 돌파 시그널
# -----------------------------
def detect_wedge_breakout(
    df: pd.DataFrame,
    pivot_left=3, pivot_right=3,     # 피벗 민감도
    K=5,                              # 추세선 적합 피벗 개수
    window=120,                       # 패턴 검증 구간
    contraction_ratio=0.65,           # 폭 축소 비율 임계치
    slope_ratio_max=0.7,              # 상/하단 기울기 비율 |lower|/|upper|<= 이 값 → 수렴(쐐기) 조건 강화
    min_touches=3,                    # 각 추세선 최소 터치(피벗) 횟수
    buffer=0.001,                     # 돌파 버퍼
    atr_period=14,
    rr_target=2.0,                    # R:R
    hold_bars=60                      # 타임아웃
):
    """
    반환: DataFrame
      columns = [upper, lower, in_wedge, wedge_type, long_entry, short_entry, long_exit, short_exit, stop_price, tp_price]
      wedge_type: 'falling_wedge'(강세), 'rising_wedge'(약세), np.nan
    입력 df: ['Open','High','Low','Close']
    """
    df = df.copy()
    Hflag, Lflag = find_pivots(df['High'], df['Low'], nL=pivot_left, nR=pivot_right)

    Hvals = pd.Series(np.nan, index=df.index); Hvals[Hflag] = df['High'][Hflag]
    Lvals = pd.Series(np.nan, index=df.index); Lvals[Lflag] = df['Low'][Lflag]

    upper = pd.Series(np.nan, index=df.index)
    lower = pd.Series(np.nan, index=df.index)
    upslope = pd.Series(np.nan, index=df.index)
    lowslope = pd.Series(np.nan, index=df.index)
    up_touches = pd.Series(0, index=df.index, dtype="int32")
    low_touches = pd.Series(0, index=df.index, dtype="int32")

    # 이동 윈도우로 추세선 계산
    for i in range(window, len(df)):
        seg_idx = df.index[i-window:i+1]
        Hseg = Hvals.loc[seg_idx]
        Lseg = Lvals.loc[seg_idx]

        uslope, uinter, uf = fit_trendline(Hseg, seg_idx, K=K)
        lslope, linter, lf = fit_trendline(Lseg, seg_idx, K=K)

        upper.iloc[i] = uf.iloc[-1] if uf.notna().any() else np.nan
        lower.iloc[i] = lf.iloc[-1] if lf.notna().any() else np.nan
        upslope.iloc[i] = uslope
        lowslope.iloc[i] = lslope

        # 터치 횟수(피벗 수) 기록
        up_touches.iloc[i] = Hseg.dropna().shape[0]
        low_touches.iloc[i] = Lseg.dropna().shape[0]

    span = (upper - lower)
    span_initial = span.shift(window-1)
    span_final = span

    # 공통 수렴 조건(폭 축소)
    contracting = (span_final <= span_initial * contraction_ratio)

    # 쐐기 판단:
    #   하락쐐기: 상/하단 모두 음의 기울기(우하향), 단 하단 기울기 절댓값이 상단보다 덜 가파름(|low| < |up|) + 슬로프비율 제한
    #   상승쐐기: 상/하단 모두 양의 기울기(우상향), 단 상단 기울기 절댓값이 하단보다 덜 가파름(|up| < |low|)
    valid_touches = (up_touches >= min_touches) & (low_touches >= min_touches)

    both_down = (upslope < 0) & (lowslope < 0)
    both_up   = (upslope > 0) & (lowslope > 0)

    # 기울기 비율 체크(쐐기 특유의 비대칭 수렴 강화)
    slope_ratio = (lowslope.abs() / upslope.abs())
    falling_wedge = valid_touches & contracting & both_down & (slope_ratio <= slope_ratio_max)  # |low| <= ratio*|up|
    # 상승쐐기는 반대로 상단이 덜 가파름 → |up| <= ratio*|low|
    slope_ratio_ru = (upslope.abs() / lowslope.abs())
    rising_wedge = valid_touches & contracting & both_up & (slope_ratio_ru <= slope_ratio_max)

    wedge_type = pd.Series(np.nan, index=df.index, dtype="object")
    wedge_type[falling_wedge] = 'falling_wedge'
    wedge_type[rising_wedge] = 'rising_wedge'

    # 내부 체류(가격이 상·하단 사이)
    inside = (df['Close'] <= upper*(1+buffer)) & (df['Close'] >= lower*(1-buffer))
    in_wedge = (falling_wedge | rising_wedge) & inside

    # 돌파(직전엔 내부, 이번 봉에 상단/하단 돌파)
    prev_inside = in_wedge.shift(1).fillna(False)
    break_up = prev_inside & (df['Close'] > upper*(1+buffer))
    break_dn = prev_inside & (df['Close'] < lower*(1-buffer))

    # 기대 방향:
    #   하락쐐기 → 상향 돌파 bias(롱)
    #   상승쐐기 → 하향 돌파 bias(숏)
    long_entry  = break_up  & (wedge_type.shift(1) == 'falling_wedge')
    short_entry = break_dn  & (wedge_type.shift(1) == 'rising_wedge')

    # 리스크/목표
    df['ATR'] = atr(df, period=atr_period)
    stop_long_line  = lower
    stop_short_line = upper
    # 엔트리 시점 기준 고정값 사용(백테스트 명확성)
    rr = rr_target

    long_exit  = pd.Series(False, index=df.index)
    short_exit = pd.Series(False, index=df.index)
    stop_price = pd.Series(np.nan, index=df.index)
    tp_price   = pd.Series(np.nan, index=df.index)

    def walkout(entry_mask, is_long=True):
        exits = pd.Series(False, index=df.index)
        sp = pd.Series(np.nan, index=df.index)
        tp = pd.Series(np.nan, index=df.index)

        entries = list(np.where(entry_mask)[0])
        for ei in entries:
            end_i = min(ei + hold_bars, len(df)-1)
            # 엔트리 기준가(여기서는 돌파 봉 종가 가정)
            entry = df['Close'].iloc[ei]
            atr_e = df['ATR'].iloc[ei]
            if np.isnan(atr_e) or atr_e == 0:
                continue

            if is_long:
                sj = min(entry - atr_e, stop_long_line.iloc[ei]) if not np.isnan(stop_long_line.iloc[ei]) else entry - atr_e
                tj = entry + rr * atr_e
                for j in range(ei+1, end_i+1):
                    # 동시 충돌시 보수적으로 손절 우선
                    if df['Low'].iloc[j] <= sj:
                        exits.iloc[j] = True; sp.iloc[j] = sj; break
                    if df['High'].iloc[j] >= tj:
                        exits.iloc[j] = True; tp.iloc[j] = tj; break
                else:
                    exits.iloc[end_i] = True
            else:
                sj = max(entry + atr_e, stop_short_line.iloc[ei]) if not np.isnan(stop_short_line.iloc[ei]) else entry + atr_e
                tj = entry - rr * atr_e
                for j in range(ei+1, end_i+1):
                    if df['High'].iloc[j] >= sj:
                        exits.iloc[j] = True; sp.iloc[j] = sj; break
                    if df['Low'].iloc[j] <= tj:
                        exits.iloc[j] = True; tp.iloc[j] = tj; break
                else:
                    exits.iloc[end_i] = True
        return exits, sp, tp

    le, lsp, ltp = walkout(long_entry,  True)
    se, ssp, stp = walkout(short_entry, False)

    long_exit  |= le; short_exit |= se
    stop_price.update(lsp.fillna(method='ffill')); stop_price.update(ssp.fillna(method='ffill'))
    tp_price.update(ltp.fillna(method='ffill'));   tp_price.update(stp.fillna(method='ffill'))

    out = pd.DataFrame({
        'upper': upper, 'lower': lower,
        'in_wedge': in_wedge,
        'wedge_type': wedge_type,
        'long_entry': long_entry, 'short_entry': short_entry,
        'long_exit': long_exit, 'short_exit': short_exit,
        'stop_price': stop_price, 'tp_price': tp_price
    }, index=df.index)
    return out

# -----------------------------
# 예시
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')  # ['Open','High','Low','Close']
    # sig = detect_wedge_breakout(df)
    # print(sig.tail(30))
    pass
