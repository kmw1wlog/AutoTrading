# 주식 기본 지표 분석
# ------------------------------------------------------------
import pandas as pd
import json
import sys
import asyncio
from typing import Literal
from pathlib import Path
import asyncio
from datetime import datetime

# project 루트 디렉토리를 파이썬 path에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from config.config_loader import openai_client
from data_loader import FinancialsFetcher, ChartFetcher

# 보고서 매니저 클래스(최신 조회, 업데이트 필요성 확인, 보고서 저장 )
from analysis.report_manager import ReportManager

# kis 모듈 임포트 추가
from kis.kis import Kis

# 클래스 구조
class FundamentalAnalysis:
    def __init__(self):
        # analysis_settings.json 파일 로드
        settings_path = Path(__file__).parent / "analysis_settings.json"
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # ReportManager 인스턴스화
        self.report_manager = ReportManager(
            report_root=project_root / "report",
            report_type="fundamental",  # 폴더 구분용
            settings=settings["fundamental"],  # update_interval이 들어있는 딕셔너리
        )

        # 데이터 로더 인스턴스 생성
        self.financials_fetcher = FinancialsFetcher()
        self.chart_fetcher = ChartFetcher()

        # Kis 인스턴스 추가
        self.kis = Kis()

        # 재무제표 캐시
        # 키: (market, stock_code) 튜플
        # 값: (재무제표 데이터: pd.DataFrame, 기업명: str) 튜플
        self.financials_cache = {}

        # 중간 지표 캐시
        # 키: (market, stock_code) 튜플
        # 값: 중간 지표 데이터
        self.indicators_cache = {}

    def _get_financials(
        self, market: Literal["dom", "ovs"], stock_code: str
    ) -> tuple[pd.DataFrame, str]:
        """
        재무제표 데이터 로드
        Args:
            market (str): "dom" or "ovs"
            stock_code (str): 종목 코드
        Returns:
            tuple[pd.DataFrame, str]
                (재무제표 데이터, 기업명)
        """
        # financials, corp_name 튜플 반환
        financials = self.financials_fetcher.get_financials(market, stock_code)
        corp_name = self.financials_fetcher.ticker_to_company_name(market, stock_code)
        self.financials_cache[(market, stock_code)] = financials, corp_name
        return financials, corp_name

    async def get_report(
        self, market: str, stock_code: str, overwrite=False, extra_prompt=""
    ) -> str:
        """
        기본 분석 보고서 생성
        :param market: str
            "dom" or "ovs"
        :param stock_code: str
        :param overwrite: bool
            if True, overwrite the existing report
        :return: str
            report_md
        """
        # 최신 보고서 파일 확인
        latest_report = self.report_manager.get_latest_report(market, stock_code)

        if latest_report:
            latest_time, latest_file = latest_report
            if not overwrite and not self.report_manager.is_update_required(
                latest_time
            ):
                return latest_file.read_text(encoding="utf-8")
            else:
                print(f"[업데이트 필요] {stock_code} 기본 보고서 갱신 중...")
        else:
            print(f"[신규 생성] {stock_code} 기본 보고서 생성 중...")

        # ─────────────────────
        # 데이터 수집 및 분석
        # ─────────────────────

        # 재무제표 + 기업명 불러오기
        financials, corp_name = self._get_financials(market, stock_code)

        # PER, PBR 계산
        PER, PBR = self.get_indicators(market, stock_code)

        stocks = pd.read_csv(
            project_root / "data" / f"charts/{market}" / "daily" / f"{stock_code}.csv"
        )

        currency = "USD" if market == "ovs" else "KRW"

        prompt_fundamental_dev = f"""
너는 투자 분석가야. 다음 기업 {corp_name}의 연간 사업보고서, 추출된 재무 지표, 최근 주가 차트를 참고하여 이 기업의 재무 상태를 요약, 분석한 Markdown 형식의 보고서를 한국어로 출력해줘.
직접적인 투자 의견을 제시하는 것은 피하고, 기업의 최근 재무 상태 분석에 초점을 맞춰줘.

보고서 본문은 주어진 JSON 구조에 맞추고, 다음 섹션들을 포함해줘.
<report_structure>
- 종합 요약: 핵심 재무 상태와 추세
- 주요 재무지표 분석: PER, PBR의 의미와 테슬라의 현재 상태
- 재무제표 분석:
  * 손익계산서 분석: 매출, 비용, 이익률 추세
  * 재무 건전성 분석: 자산, 부채, 자본 구조
  * 현금흐름 분석: 영업, 투자, 재무활동 현금흐름
- 결론 및 투자 시사점
</report_structure>
        
        {extra_prompt}
        """
        prompt_fundamental_user = f"""
        최근 연간 사업보고서: {financials.to_json(indent=4, force_ascii=False)}
        PBR = {PBR:.2f}, PER = {PER:.2f}
        최근 일별 주가 정보({currency})
        """

        fund_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "fundamental_report",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intro": {"type": "string"},
                        "business_report": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "heading": {"type": "string"},
                                    "content": {"type": "string"},
                                },
                                "required": ["heading", "content"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["intro", "business_report"],
                    "additionalProperties": False,
                },
            },
        }

        response = openai_client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "developer", "content": prompt_fundamental_dev},
                {
                    "role": "user",
                    "content": prompt_fundamental_user
                    + stocks.to_json(indent=2, force_ascii=False),
                },
            ],
            response_format=fund_response_format,
            reasoning_effort="low",
        )

        fund_report = json.loads(response.choices[0].message.content)

        # ─────────────────────
        # Markdown 변환
        # ─────────────────────
        report_md = "\n"
        report_md += f"{fund_report['intro']}\n"
        report_md += "\n---\n\n"

        for item in fund_report["business_report"]:
            item_heading = item["heading"]
            item_content = item["content"]
            report_md += f"### [{item_heading}]\n\n"
            report_md += f"{item_content}\n\n"
            report_md += "---\n\n"

        # ─────────────────────
        # 저장 및 리턴
        # ─────────────────────
        report_path = self.report_manager.save_report(market, stock_code, report_md)
        print(f"💾 기본 보고서가 생성되었습니다. \n📁 저장 경로: {report_path}\n\n")
        return report_md

    # PER(주가 수익비율), PBR(주가 순자산 비율) 반환
    def get_indicators(self, market: str, stock_code: str) -> tuple[float, float]:
        """
        KIS에서 PER, PBR 바로 가져오기
        캐시에 있으면 캐시에서 지표 반환, 없으면 지표 계산하고 캐시에 저장

        :param market: str ("dom" | "ovs")
        :param stock_code: str
        :return: PER, PBR
        """
        # 1. 캐시 확인 -> 있으면 바로 반환
        if (market, stock_code) in self.indicators_cache:
            cached = self.indicators_cache[(market, stock_code)]
            # print(f'cache hit: {cached}')
            return cached["PER"], cached["PBR"]

        # 데이터 조회(kis 모듈 활용)
        if market == "dom":
            df = self.kis.get_dom_detail(stock_code)
            PER = float(df["per"].iloc[0])
            PBR = float(df["pbr"].iloc[0])
        else:
            df = self.kis.get_overseas_detail(stock_code)
            PER = float(df["PER"].iloc[0])
            PBR = float(df["PBR"].iloc[0])

        # 캐시에 저장
        self.indicators_cache[(market, stock_code)] = {"PER": PER, "PBR": PBR}

        return PER, PBR

    async def get_indicators_table_data(self, market, stock_code):
        """
        지표 테이블 데이터 반환
        Args:
            market (str): "dom" | "ovs"
            stock_code (str): 종목 코드
        Returns:
            pd.DataFrame: 지표 테이블 데이터
        """
        PER, PBR = self.get_indicators(market, stock_code)
        # 배당수익률 (해외는 지원안함)
        # dividend_yield = 0.40
        data = pd.DataFrame(
            {
                "지표": ["PER(주가수익비율)", "PBR(주가순자산비율)"],
                "값": [PER, PBR],
            }
        )
        data.set_index("지표", inplace=True)
        return data

    async def get_indicators_graph_data(
        self, market: str, stock_code: str
    ) -> pd.DataFrame:
        """
        기본 지표 그래프용 데이터 반환 (EPS, BPS, ROE, 영업이익률)
        - 국내 / 해외 구분해서 KIS API 호출
        - 연도를 인덱스로 변환한 DataFrame 반환

        Args:
            market (str): "dom" | "ovs"
            stock_code (str): 종목 코드

        Returns:
            pd.DataFrame: 인덱스가 연도이고, 컬럼은 지표 데이터인 DF
        """

        # ─────────────────────
        # 국내 / 해외 분기 처리
        # ─────────────────────
        if market == "dom": # 국내 종목
            df = self.kis.get_dom_financial_ratio(stock_code)
            if df.empty:
                raise ValueError(f"국내 재무비율 데이터 없음: {stock_code}")
            
            # EPS, BPS만 추출, 컬럼명 변환
            df["EPS"] = df["EPS"].astype(float)
            df["BPS"] = df["BPS"].astype(float)

            # df.index = df.index.map(str)
            # yyyymm -> yyyy-mm 형식으로 변환
            df["Date"] = df["stac_yymm"].apply(lambda x: f"{x[:4]}-{x[4:]}")
            # 3년 전 데이터까지 자르기
            df = df.sort_values(by="Date", ascending=True)
            three_years_ago = (datetime.now() - pd.DateOffset(years=3)).strftime("%Y-%m")
            df = df[df["Date"] >= three_years_ago]
            df.set_index("Date", inplace=True)

        elif market == "ovs": # 해외 종목
            # 재무제표 데이터 로드
            df = self.financials_fetcher.get_financials(market, stock_code)
            if df.empty:
                raise ValueError(f"해외 재무비율 데이터 없음: {stock_code}")
            
            # Metric 컬럼에서 필요한 데이터 추출
            temp_df = df[df["Metric"] == "EarningsPerShareDiluted"][["Date", "Value"]].rename(columns={"Value": "EPS"})
            temp_df2 = df[df["Metric"] == "StockholdersEquity"][["Date", "Value"]].rename(columns={"Value": "Equity"})
            temp_df3 = df[df["Metric"] == "CommonStockSharesOutstanding"][["Date", "Value"]].rename(columns={"Value": "Shares"})
            
            # 데이터 병합
            df = pd.merge(temp_df, temp_df2, on="Date", how="left")
            df = pd.merge(df, temp_df3, on="Date", how="left")
            
            # BPS 계산
            df["BPS"] = round(df["Equity"] / df["Shares"], 2)

            # 컬럼명 포매팅
            df.set_index("Date", inplace=True)
            df.index = df.index.map(lambda x: f"{x[:7]}")

        else:
            raise ValueError(f"지원하지 않는 시장 구분: {market}")

        # ─────────────────────
        # 최종 반환
        # ─────────────────────
        # return df[["EPS($)", "BPS($)", "ROE(%)", "영업이익률(%)"]]
        return df[["EPS", "BPS"]]

if __name__ == "__main__":
    fund = FundamentalAnalysis()
    # 국내 종목 테스트
    market = "dom"
    stock_code = "005930"
    # test get_indicators_graph_data in synchronous mode
    print(asyncio.run(fund.get_indicators_graph_data(market, stock_code)))

    # 해외 종목 테스트
    market = "ovs"
    stock_code = "AAPL"
    print(asyncio.run(fund.get_indicators_graph_data(market, stock_code)))
    stock_code = "TSLA"
    print(asyncio.run(fund.get_indicators_graph_data(market, stock_code)))