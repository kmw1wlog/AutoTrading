# 대시보드 메인 파일
import streamlit as st
import asyncio
from datetime import datetime, timedelta
from analysis import FundamentalAnalysis, TechnicalAnalysis, NewsAnalysis
from data_loader.financials import FinancialsFetcher  # 종목코드 체크용
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# 페이지 설정
st.set_page_config(layout="wide", page_title="📊 주식 데이터 분석 대시보드")

# 커스텀 CSS 추가
st.markdown(
    """
<style>
.dashboard-container {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 15px;
    margin-bottom: 15px;
    background-color: black;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.scrollable-report {
    max-height: 40vh;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #eee;
    border-radius: 5px;
    background-color: #1e1e1e; /* 더 짙은 회색 배경색 (다크모드 기준) */
}
.subheader-custom {
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-top: 1px solid #ddd;
}
</style>
""",
    unsafe_allow_html=True,
)

# 인스턴스 생성
tech_analysis = TechnicalAnalysis()
fund_analysis = FundamentalAnalysis()
news_analysis = NewsAnalysis()

# 세션 스테이트 초기화
# fetched: 데이터 조회 여부
if "fetched" not in st.session_state:
    st.session_state["fetched"] = False
# chart_data: 차트 데이터
if "chart_data" not in st.session_state:
    st.session_state["chart_data"] = None
# fund_data: 기본 지표 데이터
if "fund_data" not in st.session_state:
    st.session_state["fund_data"] = None
# tech_data: 기술적 분석 데이터
if "tech_data" not in st.session_state:
    st.session_state["tech_data"] = None
# news_data: 뉴스 기사 데이터
if "news_data" not in st.session_state:
    st.session_state["news_data"] = None

# ─────────────────────────────────────────────────────────────
# 사이드바: 영역 비율 조절 슬라이더
st.sidebar.subheader("레이아웃 비율 조절")
main_left_ratio = st.sidebar.slider("메인 콘텐츠 좌측 영역 비율", 0.1, 0.9, 0.7)
main_right_ratio = 1 - main_left_ratio

st.sidebar.markdown("**기본 지표 영역 (좌측 하단) 비율**")
col1_ratio = st.sidebar.slider("기본 지표 영역 - col1 비율", 0.1, 0.9, 0.2)
col2_ratio = st.sidebar.slider("기본 지표 영역 - col2 비율", 0.1, 0.9, 0.4)
# 나머지 비율 계산 (항상 0보다 크도록 보정)
col3_ratio = max(1 - col1_ratio - col2_ratio, 0.1)

# 1) 컨트롤 영역 (container_control)
container_control = st.container()
with container_control:
    st.subheader("🔍 검색 & 컨트롤")
    market = st.selectbox("시장 선택", ["해외", "국내"])
    market = "dom" if market == "국내" else "ovs"
    stock_code = st.text_input("종목 코드 입력", "AAPL")
    date_range = st.date_input(
        "차트 조회 기간 선택", [datetime.now() - timedelta(days=180), datetime.now()]
    )

    col_btn1, col_btn2 = st.columns([0.5, 0.5])
    with col_btn1:
        try:
            company_name = FinancialsFetcher().ticker_to_company_name(
                market, stock_code
            )
        except Exception as e:
            # Create container below text input box
            error_msg_container = st.container()
            with error_msg_container:
                st.error(f"{e} 종목 코드를 다시 입력해주세요.")
            st.session_state["fetched"] = False
        async_mode = (
            True  # 비동기모드로 고정 (동기 모드에서는 너무오래걸림)
        )
        # button for report overwrite
        overwrite = st.checkbox("보고서 덮어쓰기", value=False)
        if st.button("데이터 조회"):
            st.session_state["fetched"] = True
            with st.spinner("데이터를 가져오는 중..."):
                if async_mode:

                    async def fetch_all_data():

                        # 재시도를 위한 메서드 (람다 함수 넘기기)
                        warning_placeholder = st.empty()  # 경고창 자리 만들기

                        async def fetch_with_retry(
                            name, coro_func, retries=3, delay=1, extra_prompt=None
                        ):
                            for attempt in range(retries):
                                try:
                                    result = await coro_func(extra_prompt)
                                    if result is None:
                                        warning_placeholder.warning(
                                            f"❌ {name} 데이터를 가져오지 못했습니다. None 반환됨."
                                        )
                                        raise ValueError("Result is None")

                                    warning_placeholder.empty()  # 성공 시 경고창 제거!
                                    return result
                                except Exception as e:
                                    warning_placeholder.warning(
                                        f"⚠️ {name} 실패 {attempt + 1}/{retries}: {e}"
                                    )
                                    print(f"⚠️ {name} 실패 {attempt + 1}/{retries}: {e}")

                                    if (
                                        "JSON" in str(e)
                                        or "schema" in str(e)
                                        or attempt >= 0
                                    ):
                                        extra_prompt = """
                                        반드시 JSON 스키마를 정확하게 지키세요!
                                        - 필수 필드는 모두 포함할 것
                                        - 필드명 오류, 오타 금지
                                        - 스키마 외 항목 추가 금지
                                        """

                                    await asyncio.sleep(delay)

                            return None

                        tasks = [
                            # 기술적 분석 차트 (extra_prompt 필요 없음 → 기본 람다)
                            asyncio.create_task(
                                fetch_with_retry(
                                    "기술적 분석 차트",
                                    lambda _: tech_analysis.get_rich_chart(
                                        market, stock_code, date_range
                                    ),
                                )
                            ),
                            # 기본 지표 테이블
                            asyncio.create_task(
                                fetch_with_retry(
                                    "기본 지표 테이블",
                                    lambda _: fund_analysis.get_indicators_table_data(
                                        market, stock_code
                                    ),
                                )
                            ),
                            # 기본 지표 그래프
                            asyncio.create_task(
                                fetch_with_retry(
                                    "기본 지표 그래프",
                                    lambda _: fund_analysis.get_indicators_graph_data(
                                        market, stock_code
                                    ),
                                )
                            ),
                            # 기본적 분석 보고서 (extra_prompt 사용)
                            asyncio.create_task(
                                fetch_with_retry(
                                    "기본적 분석 보고서",
                                    lambda extra_prompt: fund_analysis.get_report(
                                        market,
                                        stock_code,
                                        overwrite,
                                        extra_prompt=extra_prompt,  # 전달
                                    ),
                                )
                            ),
                            # 기술적 분석 보고서 (extra_prompt 사용)
                            asyncio.create_task(
                                fetch_with_retry(
                                    "기술적 분석 보고서",
                                    lambda extra_prompt: tech_analysis.get_report(
                                        market,
                                        stock_code,
                                        overwrite,
                                        extra_prompt=extra_prompt,  # 전달
                                    ),
                                )
                            ),
                            # 뉴스 분석 보고서 (extra_prompt 사용)
                            asyncio.create_task(
                                fetch_with_retry(
                                    "뉴스 분석 보고서",
                                    lambda extra_prompt: news_analysis.get_report(
                                        market,
                                        stock_code,
                                        overwrite,
                                        extra_prompt=extra_prompt,  # 전달
                                    ),
                                )
                            ),
                        ]

                        # 모든 태스크 동시 실행 및 결과 수집
                        results = await asyncio.gather(*tasks)
                        return results

                    # 비동기 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # 순서 유지해서 결과 가져오기
                    (
                        chart,
                        fund_indicators_table_data,
                        fund_indicators_graph_data,
                        fund_report,
                        tech_report,
                        news_report,
                    ) = loop.run_until_complete(fetch_all_data())
                    loop.close()
                else:
                    # 인스턴스 메서드 호출
                    chart = tech_analysis.get_rich_chart(market, stock_code, date_range)
                    fund_indicators_table_data = (
                        fund_analysis.get_indicators_table_data(market, stock_code)
                    )
                    fund_indicators_graph_data = (
                        fund_analysis.get_indicators_graph_data(market, stock_code)
                    )
                    fund_report = fund_analysis.get_report(market, stock_code)
                    tech_report = tech_analysis.get_report(market, stock_code)
                    news_report = news_analysis.get_report(market, stock_code)

            # 세션 상태에 데이터 저장 - 각 데이터 유형에 맞게 수정
            st.session_state["chart_data"] = chart
            st.session_state["fund_data"] = {
                "table": fund_indicators_table_data,
                "graph": fund_indicators_graph_data,
                "report": fund_report,
            }
            st.session_state["tech_data"] = tech_report
            st.session_state["news_data"] = news_report
    with col_btn2:
        if st.button("데이터 초기화"):
            st.session_state["fetched"] = False
            st.session_state["chart_data"] = None
            st.session_state["fund_data"] = None
            st.session_state["tech_data"] = None
            st.session_state["news_data"] = None
            st.success("데이터를 초기화했습니다.")

# ─────────────────────────────────────────────────────────────
# 2) 메인 콘텐츠 영역 (container_content)
container_content = st.container()
with container_content:
    # 메인 영역 좌측, 우측 비율은 사용자가 조절한 값 사용
    col_left, col_right = st.columns([main_left_ratio, main_right_ratio])

    # ── 좌측 영역 ─────────────────────────────
    with col_left:
        # 2-1) 차트 컨테이너
        container_chart = st.container()
        with container_chart:
            st.markdown(
                '<h3 class="subheader-custom">📈 차트</h3>', unsafe_allow_html=True
            )
            if not st.session_state["fetched"]:
                st.empty().markdown("데이터를 조회해주세요.")
            else:
                # 차트 데이터 및 표시 로직은 그대로 유지
                # ... 차트 코드 유지 ...
                # Plotly 차트 코드는 변경 없이 그대로 유지
                chart_data = st.session_state["chart_data"]

                # Plotly 서브플롯 생성 (가격 차트와 거래량 차트)
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                )

                # 캔들스틱 추가 (상승봉: 빨간색, 하락봉: 파란색)
                fig.add_trace(
                    go.Candlestick(
                        x=chart_data.index,
                        open=chart_data["open"],
                        high=chart_data["high"],
                        low=chart_data["low"],
                        close=chart_data["close"],
                        increasing_line_color="red",
                        decreasing_line_color="blue",
                        name="가격",
                    ),
                    row=1,
                    col=1,
                )

                # 이동평균선 추가
                if "MA5" in chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data.index,
                            y=chart_data["MA5"],
                            mode="lines",
                            name="MA5",
                            line=dict(color="purple", width=1),
                        ),
                        row=1,
                        col=1,
                    )
                if "MA20" in chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data.index,
                            y=chart_data["MA20"],
                            mode="lines",
                            name="MA20",
                            line=dict(color="orange", width=1),
                        ),
                        row=1,
                        col=1,
                    )
                if "MA60" in chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data.index,
                            y=chart_data["MA60"],
                            mode="lines",
                            name="MA60",
                            line=dict(color="green", width=1),
                        ),
                        row=1,
                        col=1,
                    )
                if "MA120" in chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data.index,
                            y=chart_data["MA120"],
                            mode="lines",
                            name="MA120",
                            line=dict(color="brown", width=1),
                        ),
                        row=1,
                        col=1,
                    )

                # 거래량 바 차트 추가
                colors = [
                    "red" if row["close"] >= row["open"] else "blue"
                    for _, row in chart_data.iterrows()
                ]
                fig.add_trace(
                    go.Bar(
                        x=chart_data.index,
                        y=chart_data["volume"],
                        marker_color=colors,
                        name="거래량",
                    ),
                    row=2,
                    col=1,
                )

                # 차트 레이아웃 설정
                currency_unit = "$" if market == "ovs" else "₩"
                fig.update_layout(
                    title="주가 차트",
                    xaxis_title="날짜",
                    yaxis_title=f"가격 ({currency_unit})",
                    xaxis_rangeslider_visible=False,
                    height=440,
                    template="plotly_white",
                    hovermode="x unified",
                )

                # Y축 타이틀 설정
                fig.update_yaxes(title_text=f"가격 ({currency_unit})", row=1, col=1)
                fig.update_yaxes(title_text="거래량", row=2, col=1)

                # X축 설정
                data_length = len(chart_data)
                if data_length <= 30:
                    tick_interval = 5  # 적은 데이터는 5일 간격
                elif data_length <= 90:
                    tick_interval = 10  # 3개월 정도는 10일 간격
                elif data_length <= 180:
                    tick_interval = 15  # 6개월 정도는 15일 간격
                elif data_length <= 365:
                    tick_interval = 30  # 1년 정도는 월 간격
                else:
                    tick_interval = 60  # 1년 이상은 2개월 간격

                # 표시할 날짜 인덱스 선택
                tick_indices = list(range(0, data_length, tick_interval))
                # 마지막 데이터 포인트가 포함되지 않았다면 추가
                if data_length - 1 not in tick_indices:
                    tick_indices.append(data_length - 1)

                # 선택된 인덱스에 해당하는 날짜만 표시
                selected_dates = [chart_data.index[i] for i in tick_indices]

                fig.update_xaxes(
                    type="category",  # 카테고리 타입으로 설정하여 실제 데이터 포인트만 표시
                    tickmode="array",
                    tickvals=selected_dates,  # 선택된 날짜만 눈금으로 표시
                    tickangle=30,  # 날짜 레이블 기울이기
                    row=1,
                    col=1,
                )
                fig.update_xaxes(
                    type="category",
                    tickmode="array",
                    tickvals=selected_dates,
                    tickangle=30,
                    row=2,
                    col=1,
                )

                # 차트 표시
                st.plotly_chart(fig, use_container_width=True)

        # 2-2) 기본 지표 영역 (컨테이너 내에 3개 컬럼)
        container_fund = st.container()
        with container_fund:
            st.markdown(
                '<h3 class="subheader-custom">📊 기본 지표 & AI 보고서 - 기본적 분석</h3>',
                unsafe_allow_html=True,
            )
            col_fund_1, col_fund_2, col_fund_3 = st.columns(
                [col1_ratio, col2_ratio, col3_ratio]
            )
            with col_fund_1:
                # st.caption("기본 지표 테이블")
                if not st.session_state["fetched"]:
                    st.empty().markdown("데이터를 조회해주세요.")
                else:
                    # 테이블 데이터를 예쁘게 표시
                    table_data = st.session_state["fund_data"]["table"]
                    st.dataframe(
                        table_data,
                        use_container_width=True,
                        hide_index=False,
                        column_config={
                            "값": st.column_config.NumberColumn(
                                format="%.2f", help="지표 값"
                            )
                        },
                    )
            with col_fund_2:
                # st.caption("기본 지표 그래프")
                if not st.session_state["fetched"]:
                    st.empty().markdown("데이터를 조회해주세요.")
                else:
                    # 그래프 데이터를 예쁜 꺾은선 그래프로 표시 - 코드 유지
                    graph_data = st.session_state["fund_data"]["graph"]

                    # 데이터 유효성 검사 추가!
                    if graph_data is None or graph_data.empty:
                        st.warning(
                            "기본 지표 그래프 데이터가 없습니다. 다시 조회해주세요."
                        )
                    else:
                        # 달러 단위와 퍼센트 단위를 분리하여 두 개의 y축 사용
                        fig = make_subplots(specs=[[{"secondary_y": True}]])

                    # 각 지표별로 선 추가 (단위에 따라 y축 구분)
                    for column in graph_data.columns:
                        if column in ["BPS"]:  # BPS는 주 y축
                            fig.add_trace(
                                go.Scatter(
                                    x=graph_data.index,
                                    y=graph_data[column],
                                    mode="lines+markers",
                                    name=column,
                                ),
                                secondary_y=False,
                            )
                        elif column in ["EPS"]:  # EPS는 보조 y축
                            fig.add_trace(
                                go.Scatter(
                                    x=graph_data.index,
                                    y=graph_data[column],
                                    mode="lines+markers",
                                    name=column,
                                ),
                                secondary_y=True,
                            )

                    # y축 범위 설정 - 여유 있게 설정
                    # 1. 주 y축: BPS
                    bps_values = graph_data[["BPS"]].values.flatten()
                    bps_values = [
                        v for v in bps_values if v is not None and pd.notna(v)
                    ]

                    if len(bps_values) > 0:
                        bps_max = max(bps_values) * 1.5
                    else:
                        bps_max = 50000 if currency_unit == "₩" else 100  # 기본값 설정

                    fig.update_yaxes(range=[0, bps_max], secondary_y=False)

                    # 보조 y축: EPS 스케일
                    eps_values = graph_data[["EPS"]].values.flatten()

                    # None, NaN 제거
                    eps_values = [
                        v for v in eps_values if v is not None and pd.notna(v)
                    ]

                    if len(eps_values) > 0:
                        eps_max = max(eps_values) * 1.2
                    else:
                        eps_max = (
                            1000 if currency_unit == "₩" else 1
                        )  # 기본값 (원하는 값으로 조정 가능)

                    fig.update_yaxes(range=[0, eps_max], secondary_y=True)
                    # 그래프 레이아웃 설정
                    fig.update_layout(
                        xaxis_title="연도",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                        hovermode="x unified",
                    )

                    # y축 제목 설정
                    fig.update_yaxes(
                        title_text=f"BPS ({currency_unit})", secondary_y=False
                    )
                    fig.update_yaxes(
                        title_text=f"EPS ({currency_unit})", secondary_y=True
                    )

                    # 그래프 표시
                    # BPS는 주 y축, EPS는 보조 y축
                    fig.update_layout(
                        yaxis2=dict(
                            title_text=f"EPS ({currency_unit})",
                            overlaying="y",
                            side="right",
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col_fund_3:
                # st.caption("AI 보고서 - 기본적 분석")
                if not st.session_state["fetched"]:
                    st.empty().markdown("데이터를 조회해주세요.")
                else:
                    # 스크롤 가능한 박스에 보고서 표시
                    st.markdown(
                        f'<div class="scrollable-report">{st.session_state["fund_data"]["report"]}</div>',
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── 우측 영역 ─────────────────────────────
    with col_right:
        # 2-3) 기술적 분석 컨테이너
        container_tech = st.container()
        with container_tech:
            st.markdown(
                '<h3 class="subheader-custom">🛠️ AI 보고서 - 기술적 분석</h3>',
                unsafe_allow_html=True,
            )
            if not st.session_state["fetched"]:
                st.empty().markdown("데이터를 조회해주세요.")
            else:
                # 스크롤 가능한 박스에 보고서 표시 (높이 45vh로 설정)
                st.markdown(
                    f'<div class="scrollable-report" style="max-height: 45vh;">{st.session_state["tech_data"]}</div>',
                    unsafe_allow_html=True,
                )
                # st.markdown(st.session_state["tech_data"])
        st.markdown("</div>", unsafe_allow_html=True)

        # 2-4) 뉴스 기사 컨테이너
        container_news = st.container()
        with container_news:
            st.markdown(
                '<h3 class="subheader-custom">📰 AI 보고서 - 뉴스 기사</h3>',
                unsafe_allow_html=True,
            )
            if not st.session_state["fetched"]:
                st.empty().markdown("데이터를 조회해주세요.")
            else:
                # 스크롤 가능한 박스에 보고서 표시
                st.markdown(
                    f'<div class="scrollable-report style="max-height: 45vh;"">{st.session_state["news_data"]}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)