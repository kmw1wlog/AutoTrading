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
# 선형 보간 넥라인 함수 (두 점)
# -----------------------------
def line_value(x0, y0, x1, y1, xq):
    # x는 정수 인덱스 축 (bar index)
    if x1 == x0:
        return y0
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

# -----------------------------
# 헤드앤숄더 / 역헤드앤숄더 탐지 + 돌파 시그널
# -----------------------------
def detect_head_and_shoulders(
    df: pd.DataFrame,
    pivot_left=3, pivot_right=3,   # 피벗 민감도
    lookback=250,                  # 패턴 탐지에 사용할 최대 과거 구간
    shoulder_tol=0.15,             # 좌·우견 높이(또는 깊이) 유사성 허용 비율
    min_head_gain=0.02,            # 머리가 넥라인 대비 더 높거나(혹은 낮거나) 차이가 이 이상이어야 함 (2%)
    max_shoulder_span=60,          # 좌견~우견 전체 길이 제한(봉수)
    min_between=3,                 # 각 피벗 사이 최소 봉수
    buffer=0.001,                  # 돌파 버퍼(0.1%)
    atr_period=14,
    use_retest=False,              # 넥라인 리테스트 후 진입할지
    rr_measure=1.0,                # 측정치(target): head-네크라인 거리의 배수
    stop_atr_mult=1.0,             # ATR 기반 여유폭
    hold_bars=80                   # 타임아웃 청산
):
    """
    반환 DataFrame 컬럼:
      ['pattern','neckline','neck_p1','neck_p2',
       'long_entry','short_entry','long_exit','short_exit',
       'stop_price','tp_price']
    pattern: 'HnS'(약세), 'InvHnS'(강세), np.nan
    """
    df = df.copy()
    idx = df.index
    n = len(df)

    # 피벗
    Hflag, Lflag = find_pivots(df['High'], df['Low'], nL=pivot_left, nR=pivot_right)
    pivH = [(i, df['High'].iloc[i]) for i in range(n) if Hflag.iloc[i]]
    pivL = [(i, df['Low' ].iloc[i]) for i in range(n) if Lflag.iloc[i]]

    # ATR
    df['ATR'] = atr(df, period=atr_period)

    # 결과 버퍼
    pattern = np.array([None]*n, dtype=object)
    neckline = np.full(n, np.nan)    # 시점별 넥라인 값(마지막 RS 시점 기준 값 저장)
    neck_p1  = np.full(n, np.nan)    # 넥라인 점1 값
    neck_p2  = np.full(n, np.nan)    # 넥라인 점2 값
    long_entry  = np.zeros(n, dtype=bool)
    short_entry = np.zeros(n, dtype=bool)
    long_exit   = np.zeros(n, dtype=bool)
    short_exit  = np.zeros(n, dtype=bool)
    stop_price  = np.full(n, np.nan)
    tp_price    = np.full(n, np.nan)

    # 도우미: 두 pivot 사이 가장 최근의 반대 pivot 찾기
    def find_between_pivot(piv_list, left_i, right_i):
        # piv_list: (bar, value) list, left_i/right_i는 bar 인덱스
        cands = [p for p in piv_list if left_i < p[0] < right_i]
        if not cands:
            return None
        # 가장 마지막(가까운) pivot을 택해도 되고, 중앙 pivot을 택해도 됨
        # 여기선 첫 번째(왼→오) 최대/최소 중 '중간 pivot'을 선택
        # 이미 piv_list가 시간 순이라 가정
        return cands[-1]

    # 스캔 전략:
    # 1) H&S(정배열, 약세): pivot highs 중 3개(h1<h2>h3)와 그 사이 pivot lows(l1, l2)
    # 2) Inverse H&S(역배열, 강세): pivot lows 중 3개(l1>l2<l3)와 그 사이 pivot highs(h1, h2)
    # 간단화를 위해 최근 lookback 내에서 윈도우 스캔
    start = max(0, n - lookback)

    # ---- H&S (약세) ----
    Hbars = [b for (b, v) in pivH]
    for i2 in range(1, len(Hbars)-1):
        h1_bar = Hbars[i2-1]
        h2_bar = Hbars[i2]
        h3_bar = Hbars[i2+1]
        if not (h1_bar < h2_bar - min_between and h2_bar < h3_bar - min_between):
            continue
        if h1_bar < start:
            continue
        # 좌견-머리-우견 구간 제한
        if h3_bar - h1_bar > max_shoulder_span:
            continue
        h1 = df['High'].iloc[h1_bar]
        h2 = df['High'].iloc[h2_bar]
        h3 = df['High'].iloc[h3_bar]
        # 머리 최고
        if not (h2 > h1 and h2 > h3):
            continue
        # 중간 저점들
        l1_p = find_between_pivot(pivL, h1_bar, h2_bar)
        l2_p = find_between_pivot(pivL, h2_bar, h3_bar)
        if l1_p is None or l2_p is None:
            continue
        l1_bar, l1 = l1_p
        l2_bar, l2 = l2_p

        # 넥라인: (l1_bar, l1) ~ (l2_bar, l2)
        # 패턴 강도: 머리-넥라인 거리
        mid_head_line = line_value(l1_bar, l1, l2_bar, l2, h2_bar)
        head_over_neck = (h2 - mid_head_line) / mid_head_line
        if head_over_neck < min_head_gain:
            continue

        # 좌·우견 유사성(넥라인 대비 고점 높이)
        sh1_over = (h1 - line_value(l1_bar, l1, l2_bar, l2, h1_bar)) / max(1e-12, line_value(l1_bar, l1, l2_bar, l2, h1_bar))
        sh3_over = (h3 - line_value(l1_bar, l1, l2_bar, l2, h3_bar)) / max(1e-12, line_value(l1_bar, l1, l2_bar, l2, h3_bar))
        if not (abs(sh1_over - sh3_over) <= shoulder_tol * max(1e-12, abs(sh1_over) + abs(sh3_over)) + 1e-12):
            continue

        # 패턴 확정 시점: 우견(h3_bar)
        # 돌파 조건: 이후 넥라인 하향 돌파
        # 먼저 넥라인 값 저장
        neckline_val_at_rs = line_value(l1_bar, l1, l2_bar, l2, h3_bar)
        neckline[h3_bar] = neckline_val_at_rs
        neck_p1[h3_bar] = l1
        neck_p2[h3_bar] = l2
        pattern[h3_bar] = 'HnS'

        # 돌파 체크: 다음 봉부터
        for j in range(h3_bar+1, min(h3_bar+hold_bars, n-1)+1):
            neck_j = line_value(l1_bar, l1, l2_bar, l2, j)
            if use_retest:
                # 1) 첫 하향 이탈 → 2) 리테스트(목선 근처 반등 실패) → 3) 재하향 시 진입
                # 간단 구현: 이탈 플래그 후, 종가가 neck_j*(1±buffer) 근접했다가 다시 종가<neck_j*(1-buffer) 시 론칭
                # 상태 머신 없이 근사:
                pass
            if df['Close'].iloc[j] < neck_j * (1 - buffer):
                short_entry[j] = True
                # 손절: 우견고점 또는 넥라인 + ATR 여유 중 높은 쪽
                entry = df['Close'].iloc[j]
                atr_e = df['ATR'].iloc[j]
                rs_high = h3
                sj = max(rs_high, neck_j) + stop_atr_mult * atr_e
                # 목표: 측정치 = (머리-넥라인) * rr_measure
                measure = (h2 - mid_head_line) * rr_measure
                tj = entry - measure
                # 이후 청산
                end_j = min(j + hold_bars, n-1)
                for k in range(j+1, end_j+1):
                    # 동시 충돌시 보수적으로 손절 우선
                    if df['High'].iloc[k] >= sj:
                        short_exit[k] = True; stop_price[k] = sj; break
                    if df['Low'].iloc[k] <= tj:
                        short_exit[k] = True;  tp_price[k] = tj;  break
                else:
                    short_exit[end_j] = True
                break  # 한 패턴에서 한 번만 진입

    # ---- Inverse H&S (강세) ----
    Lbars = [b for (b, v) in pivL]
    for i2 in range(1, len(Lbars)-1):
        l1_bar = Lbars[i2-1]
        l2_bar = Lbars[i2]
        l3_bar = Lbars[i2+1]
        if not (l1_bar < l2_bar - min_between and l2_bar < l3_bar - min_between):
            continue
        if l1_bar < start:
            continue
        if l3_bar - l1_bar > max_shoulder_span:
            continue
        l1 = df['Low'].iloc[l1_bar]
        l2 = df['Low'].iloc[l2_bar]
        l3 = df['Low'].iloc[l3_bar]
        # 머리 최저
        if not (l2 < l1 and l2 < l3):
            continue
        # 중간 고점들
        h1_p = find_between_pivot(pivH, l1_bar, l2_bar)
        h2_p = find_between_pivot(pivH, l2_bar, l3_bar)
        if h1_p is None or h2_p is None:
            continue
        h1_bar, h1 = h1_p
        h2_bar, h2 = h2_p

        # 넥라인: (h1_bar, h1) ~ (h2_bar, h2)
        mid_head_line = line_value(h1_bar, h1, h2_bar, h2, l2_bar)
        head_under_neck = (mid_head_line - l2) / max(1e-12, mid_head_line)
        if head_under_neck < min_head_gain:
            continue

        # 좌·우견 유사성(넥라인 대비 저점 깊이)
        sh1_under = (line_value(h1_bar, h1, h2_bar, h2, l1_bar) - l1) / max(1e-12, line_value(h1_bar, h1, h2_bar, h2, l1_bar))
        sh3_under = (line_value(h1_bar, h1, h2_bar, h2, l3_bar) - l3) / max(1e-12, line_value(h1_bar, h1, h2_bar, h2, l3_bar))
        if not (abs(sh1_under - sh3_under) <= shoulder_tol * max(1e-12, abs(sh1_under) + abs(sh3_under)) + 1e-12):
            continue

        neckline_val_at_rs = line_value(h1_bar, h1, h2_bar, h2, l3_bar)
        neckline[l3_bar] = neckline_val_at_rs
        neck_p1[l3_bar] = h1
        neck_p2[l3_bar] = h2
        pattern[l3_bar] = 'InvHnS'

        for j in range(l3_bar+1, min(l3_bar+hold_bars, n-1)+1):
            neck_j = line_value(h1_bar, h1, h2_bar, h2, j)
            if df['Close'].iloc[j] > neck_j * (1 + buffer):
                long_entry[j] = True
                entry = df['Close'].iloc[j]
                atr_e = df['ATR'].iloc[j]
                rs_low = l3
                sj = min(rs_low, neck_j) - stop_atr_mult * atr_e
                measure = (mid_head_line - l2) * rr_measure
                tj = entry + measure
                end_j = min(j + hold_bars, n-1)
                for k in range(j+1, end_j+1):
                    if df['Low'].iloc[k] <= sj:
                        long_exit[k] = True; stop_price[k] = sj; break
                    if df['High'].iloc[k] >= tj:
                        long_exit[k] = True;  tp_price[k] = tj;  break
                else:
                    long_exit[end_j] = True
                break

    out = pd.DataFrame({
        'pattern': pd.Series(pattern, index=idx, dtype='object'),
        'neckline': neckline,
        'neck_p1': neck_p1,
        'neck_p2': neck_p2,
        'long_entry': long_entry, 'short_entry': short_entry,
        'long_exit':  long_exit,  'short_exit':  short_exit,
        'stop_price': stop_price, 'tp_price': tp_price
    }, index=idx)
    return out

# -----------------------------
# 예시
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_csv("ohlc.csv", parse_dates=['Date'], index_col='Date')  # ['Open','High','Low','Close']
    # sig = detect_head_and_shoulders(df)
    # print(sig.tail(80))
    pass
