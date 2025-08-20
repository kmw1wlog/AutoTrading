"""
프로젝트4 실습용 코드
"""
import pandas as pd
import streamlit as st
import asyncio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from analysis import FundamentalAnalysis, TechnicalAnalysis, NewsAnalysis
from data_loader.financials import FinancialsFetcher

# ─────────────────────────────────────────────────────────────
# 페이지 설정
def setup_page():
    """페이지 설정, 커스텀 CSS 적용"""    
    # 페이지 설정
    st.set_page_config(
        layout="wide",
        page_title="AI 주식 분석 대시보드",
        page_icon="📊",
    )

    st.html("""
    <style>
    .scrollable-report {
        max-height: 40vh;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #eee;
        border-radius: 5px;
        background-color: #1e1e1e;
    }
    .subheader-custom {
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-top: 1px solid #ddd;
    }
    </style>
    """)

def initialize_session_state():
    """세션 상태 변수 초기화"""
    if "fetched" not in st.session_state:
        st.session_state["fetched"] = False
    if "chart_data" not in st.session_state:
        st.session_state["chart_data"] = None
    if "fund_data" not in st.session_state:
        st.session_state["fund_data"] = None
    if "tech_data" not in st.session_state:
        st.session_state["tech_data"] = None
    if "news_data" not in st.session_state:
        st.session_state["news_data"] = None

# 캐시된 리소스 함수 정의
@st.cache_resource
def get_tech_analysis():
    return TechnicalAnalysis()
@st.cache_resource
def get_fund_analysis():
    return FundamentalAnalysis()
@st.cache_resource
def get_news_analysis():
    return NewsAnalysis()

# 분석 인스턴스 생성
tech_analysis = get_tech_analysis()
fund_analysis = get_fund_analysis()
news_analysis = get_news_analysis()

# ─────────────────────────────────────────────────────────────
# 데이터 동시 수집
async def fetch_all_data(market, stock_code, date_range, overwrite):
    warning_placeholder = st.empty()  # 경고창 자리 만들기 (오류 발생 시 경고 표시)

    tasks = [
        asyncio.create_task(tech_analysis.get_rich_chart(
                market, stock_code, list(date_range)
            )
        ),
        asyncio.create_task(fund_analysis.get_indicators_table_data(
                market, stock_code
            )
        ),
        asyncio.create_task(fund_analysis.get_indicators_graph_data(
                market, stock_code
            )
        ),
        asyncio.create_task(fund_analysis.get_report(
                market, stock_code, overwrite
            )
        ),
        asyncio.create_task(tech_analysis.get_report(
                market, stock_code, overwrite
            )
        ),
        asyncio.create_task(news_analysis.get_report(
                market, stock_code, overwrite
            )
        ),
    ]

    # 모든 태스크 동시 실행 및 결과 수집
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 수집한 결과에서 오류 확인
    for result in results:
        if isinstance(result, Exception):
            warning_placeholder.warning(
                f"❌ 데이터를 가져오지 못했습니다. 오류: {str(result)}"
            )
            st.session_state["fetched"] = False
            return None
        
    return results

def save_data_to_session(results):
    """세션 상태에 데이터 저장"""
    if results is None:
        return
    
    chart, fund_table, fund_graph, fund_report, tech_report, news_report = results
    
    st.session_state["chart_data"] = chart
    st.session_state["fund_data"] = {
        "table": fund_table,
        "graph": fund_graph,
        "report": fund_report,
    }
    st.session_state["tech_data"] = tech_report
    st.session_state["news_data"] = news_report


def clear_session_data():
    """세션 상태 데이터 초기화"""
    st.session_state["fetched"] = False
    st.session_state["chart_data"] = None
    st.session_state["fund_data"] = None
    st.session_state["tech_data"] = None
    st.session_state["news_data"] = None

# ─────────────────────────────────────────────────────────────
# 사이드바, 컨트롤 영역 생성 및 표시
def create_sidebar_controls():
    """레이아웃 비율 조절을 위한 사이드바 구성"""
    st.sidebar.subheader("레이아웃 비율 조절")
    main_left_ratio = st.sidebar.slider("메인 콘텐츠 좌측 영역 비율", 0.1, 0.9, 0.7)
    main_right_ratio = 1 - main_left_ratio
    
    st.sidebar.markdown("**기본 지표 영역 (좌측 하단) 비율**")
    col1_ratio = st.sidebar.slider("기본 지표 영역 - col1 비율", 0.1, 0.9, 0.2)
    col2_ratio = st.sidebar.slider("기본 지표 영역 - col2 비율", 0.1, 0.9, 0.3)
    col3_ratio = max(1 - col1_ratio - col2_ratio, 0.1)
    
    return main_left_ratio, main_right_ratio, col1_ratio, col2_ratio, col3_ratio

def create_control_section():
    """사용자 입력을 위한 컨트롤 영역 생성"""    
    st.subheader("🔍 검색 & 컨트롤")
    # 시장, 종목코드, 조회 기간 선택
    market_display = st.selectbox("시장 선택", ["국내", "해외"])
    market = "dom" if market_display == "국내" else "ovs"
    stock_code = st.text_input("종목 코드 입력", "005930")
    date_range = st.date_input(
        "차트 조회 기간 선택", 
        [datetime.now() - timedelta(days=180), datetime.now()]
    )
    # 종목 코드 유효성 검사
    try:
        company_name = FinancialsFetcher().ticker_to_company_name(market, stock_code)
    except Exception as e:
        st.error(f"{e} 종목 코드를 다시 입력해주세요.")
        st.session_state["fetched"] = False
        return market, stock_code, date_range, False
    
    # 버튼 영역 생성
    col_btn1, col_btn2 = st.columns([0.5, 0.5])
    
    with col_btn1:
        overwrite = st.checkbox("보고서 덮어쓰기", value=False)
        
        if st.button("데이터 조회"):
            st.session_state["fetched"] = True
            with st.spinner("데이터를 가져오는 중..."):
                # 비동기 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    results = loop.run_until_complete(
                        fetch_all_data(market, stock_code, date_range, overwrite)
                    )
                    save_data_to_session(results)
                finally:
                    loop.close()
    
    with col_btn2:
        if st.button("데이터 초기화"):
            clear_session_data()
            st.success("데이터를 초기화했습니다.")
    
    return market

# ─────────────────────────────────────────────────────────────
# 차트 요소 생성 및 표시
def create_candlestick_chart(chart_data, market):
    """캔들 차트 생성"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
    )
    
    # 캔들 차트 추가
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
        row=1, col=1,
    )
    
    # 이동 평균선 추가
    ma_colors = {"MA5": "purple", "MA20": "orange", "MA60": "green", "MA120": "brown"}
    for ma, color in ma_colors.items():
        if ma in chart_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data[ma],
                    mode="lines",
                    name=ma,
                    line=dict(color=color, width=1),
                ),
                row=1, col=1,
            )
    
    # 거래량 막대 그래프 추가
    volume_colors = [
        "red" if row["close"] >= row["open"] else "blue"
        for _, row in chart_data.iterrows()
    ]
    fig.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["volume"],
            marker_color=volume_colors,
            name="거래량",
        ),
        row=2, col=1,
    )
    
    return fig

def configure_chart_layout(fig, chart_data, market):
    """차트 레이아웃 설정"""
    currency_unit = "$" if market == "ovs" else "₩"
    
    fig.update_layout(
        title="주가 차트",
        xaxis_title="날짜",
        yaxis_title=f"가격 ({currency_unit})",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
        hovermode="x unified",
    )
    
    fig.update_yaxes(title_text=f"가격 ({currency_unit})", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    
    # x축 눈금 설정
    data_length = len(chart_data)
    length_interval_dict = {30: 5, 90: 10, 180: 15, 365: 30}
    tick_interval = 60
    
    for length, interval in length_interval_dict.items():
        if data_length <= length:
            tick_interval = interval
            break
    
    tick_indices = list(range(0, data_length, tick_interval))
    if data_length - 1 not in tick_indices:
        tick_indices.append(data_length - 1)
    
    selected_dates = [chart_data.index[i] for i in tick_indices]
    
    for row in [1, 2]:
        fig.update_xaxes(
            type="category",
            tickmode="array",
            tickvals=selected_dates,
            tickangle=30,
            row=row, col=1,
        )

def display_chart_section(market):
    """차트 영역 표시"""
    st.markdown('<h3 class="subheader-custom">📈 차트</h3>', unsafe_allow_html=True)
    
    if not st.session_state["fetched"]:
        st.markdown("데이터를 조회해주세요.")
        return
    
    chart_data = st.session_state["chart_data"]
    fig = create_candlestick_chart(chart_data, market)
    configure_chart_layout(fig, chart_data, market)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 기본 지표 테이블, 그래프, 기본적 분석 보고서 표시
def display_fund_table():
    """기본 지표 테이블 표시"""
    if not st.session_state["fetched"]:
        st.markdown("데이터를 조회해주세요.")
        return
    
    table_data = st.session_state["fund_data"]["table"]
    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=False,
        column_config={
            "값": st.column_config.NumberColumn(format="%.2f", help="지표 값")
        },
    )

def display_fund_graph(currency_unit):
    """기본 지표 그래프 표시"""
    if not st.session_state["fetched"]:
        st.markdown("데이터를 조회해주세요.")
        return
    
    graph_data = st.session_state["fund_data"]["graph"]
    
    # 보조 y축 그래프 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 데이터 유형 따라 트레이스 추가
    for column in graph_data.columns:
        if column in ["BPS"]:
            fig.add_trace(
                go.Scatter(
                    x=graph_data.index,
                    y=graph_data[column],
                    mode="lines+markers",
                    name=column,
                ),
                secondary_y=False,
            )
        elif column in ["EPS"]:
            fig.add_trace(
                go.Scatter(
                    x=graph_data.index,
                    y=graph_data[column],
                    mode="lines+markers",
                    name=column,
                ),
                secondary_y=True,
            )
    
    # y축 범위 설정: 최댓값의 1.2배
    # 주 y축: BPS
    bps_values = graph_data[["BPS"]].values.flatten()
    bps_values = [v for v in bps_values if v is not None and pd.notna(v)]

    if len(bps_values) > 0:
        bps_max = max(bps_values) * 1.2
    else:
        bps_max = 50000 if currency_unit == "₩" else 100  # 기본값 설정

    fig.update_yaxes(range=[0, bps_max], secondary_y=False)

    # 보조 y축: EPS 스케일
    eps_values = graph_data[["EPS"]].values.flatten()

    # None, NaN 제거
    eps_values = [v for v in eps_values if v is not None and pd.notna(v)]

    if len(eps_values) > 0:
        eps_max = max(eps_values) * 1.2
    else:
        eps_max = 1000 if currency_unit == "₩" else 1

    fig.update_yaxes(range=[0, eps_max], secondary_y=True)
    
    fig.update_layout(
        xaxis_title="연도",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    
    fig.update_yaxes(title_text=f"BPS ({currency_unit})", secondary_y=False)
    fig.update_yaxes(title_text=f"EPS ({currency_unit})", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

def display_fund_report():
    """기본적 분석 보고서 표시"""
    if not st.session_state["fetched"]:
        st.markdown("데이터를 조회해주세요.")
        return
    
    report = st.session_state["fund_data"]["report"]
    st.markdown(    
        f'<div class="scrollable-report">{report}</div>',
        unsafe_allow_html=True,
    )

def display_fundamental_section(col1_ratio, col2_ratio, col3_ratio, currency_unit):
    """기본 지표 테이블, 그래프 표시"""
    st.markdown(
        '<h3 class="subheader-custom">📊 기본 지표 & AI 보고서 - 기본적 분석</h3>',
        unsafe_allow_html=True,
    )
    
    col_fund_1, col_fund_2, col_fund_3 = st.columns([col1_ratio, col2_ratio, col3_ratio])
    
    with col_fund_1:
        display_fund_table()
    with col_fund_2:
        display_fund_graph(currency_unit)
    with col_fund_3:
        display_fund_report()

def display_technical_report():
    """기술적 분석 보고서 표시"""
    st.markdown(
        '<h3 class="subheader-custom">🛠️ AI 보고서 - 기술적 분석</h3>',
        unsafe_allow_html=True,
    )
    
    if not st.session_state["fetched"]:
        st.markdown("데이터를 조회해주세요.")
        return
    
    report = st.session_state["tech_data"]
    st.markdown(
        f'<div class="scrollable-report" style="max-height: 45vh;">{report}</div>',
        unsafe_allow_html=True,
    )

def display_news_report():
    """뉴스 분석 보고서 표시"""
    st.markdown(
        '<h3 class="subheader-custom">📰 AI 보고서 - 뉴스 기사</h3>',
        unsafe_allow_html=True,
    )
    
    if not st.session_state["fetched"]:
        st.markdown("데이터를 조회해주세요.")
        return
    
    report = st.session_state["news_data"]
    st.markdown(
        f'<div class="scrollable-report">{report}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# 메인 함수
def main():
    # 초기 설정
    setup_page()
    initialize_session_state()
    
    # (1) 사이드바: 영역 비율 조절
    main_left_ratio, main_right_ratio, col1_ratio, col2_ratio, col3_ratio = create_sidebar_controls()

    # (2) 컨트롤 영역
    market = create_control_section()
    currency_unit = "$" if market == "ovs" else "₩"
    
    # (3) 메인 콘텐츠 영역
    col_left, col_right = st.columns([main_left_ratio, main_right_ratio])
    
    # (3-1) 메인 좌측 영역
    with col_left:
        display_chart_section(market)
        display_fundamental_section(col1_ratio, col2_ratio, col3_ratio, currency_unit)
    
    # (3-2) 메인 우측 영역
    with col_right:
        display_technical_report()
        display_news_report()

if __name__ == "__main__":
    main()