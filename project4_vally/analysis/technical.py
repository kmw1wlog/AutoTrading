# AI 주식 분석: 기술적 분석
import pandas as pd
import json
import sys
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# project 루트 디렉토리를 파이썬 path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config_loader import openai_client  # ✅ config에서 불러옴

from data_loader.chart import ChartFetcher

# ReportManager 가져오기
from analysis.report_manager import ReportManager

class TechnicalAnalysis:
    def __init__(self):
        # settings.json 파일 로드
        settings_path = Path(__file__).parent / "analysis_settings.json"
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # 보고서 생성 매니저
        self.report_manager = ReportManager(
            report_root=project_root / "report",
            report_type="technical",
            settings=settings["technical"]["daily"],  # 'daily'만 전달
        )

        self.chart_fetcher = ChartFetcher()

    async def get_rich_chart(self, market, stock_code, date_range: list, freq="daily"):
        """
        차트 데이터 + 이동평균선

        freq의 기본 값을 daily로 설정.(app.py에 파라미터를 설정안해도 됨.)
        추후 확장성을 고려하여 주봉, 분봉 선택할 수 있게 메서드 내에 파라미터 설정.
        """
        # date_range 파라미터 방어: datetime으로 강제 통일
        start_date = (
            pd.to_datetime(date_range[0])
            if not isinstance(date_range[0], pd.Timestamp)
            else date_range[0]
        )
        end_date = (
            pd.to_datetime(date_range[1])
            if not isinstance(date_range[1], pd.Timestamp)
            else date_range[1]
        )

        # 날짜 정보만 비교를 위해 normalize()
        start_ts = start_date.normalize()
        end_ts = end_date.normalize()

        # 차트 데이터 로드 (180일 전부터 조회)
        chart = self.chart_fetcher.get_chart(
            market,
            stock_code,
            start_datetime=start_ts - timedelta(days=180),
            end_datetime=end_ts,
            interval=freq,
        )

        # None 체크
        if chart is None or chart.empty:
            print(f"[ERROR] 차트 데이터가 없습니다: {stock_code}")
            return None

        # 이동평균선 계산 (daily만, 5~120일)
        if freq == "daily":
            for length in [5, 20, 60, 120]:
                ma = chart.ta.sma(length=length)
                column_name = f"MA{length}"
                # 데이터 프레임일 경우에만
                chart[column_name] = (
                    ma.iloc[:, 0] if isinstance(ma, pd.DataFrame) else ma
                )

        # 인덱스도 normalize 해서 날짜 기준으로 정렬
        chart.index = chart.index.normalize()

        # 날짜 필터링
        chart_filtered = chart[(chart.index >= start_ts) & (chart.index <= end_ts)]

        # 필터링 결과 예외 처리
        if chart_filtered.empty:
            print(f"[WARN] 필터링 후 차트 데이터가 없습니다. {start_ts} ~ {end_ts}")
            return None

        return chart_filtered

    async def get_report(
        self,
        market: str,
        stock_code: str,
        overwrite=False,
        freq="daily",
        extra_prompt="",
    ) -> str:
        """
        기술적 분석 보고서 생성
        Args:
            market: 시장 구분
            stock_code: 종목 코드
            overwrite: 기존 리포트 덮어쓰기 여부
            freq: 차트 주기 (daily, weekly, min)
            extra_prompt: 오류 발생 시 재시도를 위한 추가 프롬프트

        Returns:    
            생성된 리포트
        """
        # 1. 최신 리포트 파일 확인
        latest_report = self.report_manager.get_latest_report(market, stock_code)

        if latest_report:
            latest_time, latest_file = latest_report
            if not overwrite and not self.report_manager.is_update_required(
                latest_time
            ):
                return latest_file.read_text(encoding="utf-8")
            else:
                print(f"[업데이트 필요] {stock_code} 기술적 분석 리포트 갱신 중...")

        else:
            print(f"[신규 생성] {stock_code} 기술적 분석 리포트 생성 중...")

        # 2. 차트 데이터 로드 (최근 6개월)
        start_datetime = datetime.now() - timedelta(days=180)
        end_datetime = datetime.now()

        chart_data = self.chart_fetcher.get_chart(
            market, stock_code, start_datetime, end_datetime, interval="daily"
        )

        # 3. 기술적 지표 계산
        df_indicators = self.get_technical_indicators(chart_data)

        # 4. 프롬프트 작성
        prompt_tech_dev = f"""
        너는 전문 투자 분석가야. 다음 기업의 최근 6개월 간 일봉 데이터와 기술적 지표들을 참고하여 이 기업에 대한 Markdown 형식의 기술적 분석 결과를 한국어로 출력해.
        매수/매도 전략을 직접적으로 추천하지는 말고, 투자에 참고할 만한 차트 분석 결과를 제시해줘.
        분석 결과는 투자 초보자도 이해할 수 있는 수준으로 쉽고 자세하게 설명하고, 차트에서 주목할만한 시점이 있다면 그 날짜와 정확한 수치를 근거로 들어줘.

        보고서는 주어진 JSON 구조에 맞추고, 다음 섹션들을 포함해 작성해줘.
        <report_headers>
        종합 요약
        이동평균선 분석
        주요 차트 포인트 (지지 및 저항 영역, 거래량 변화, 추세 전환 시점 등)
        결론 및 투자 시사점
        </report_headers>   
        
        {extra_prompt}
        """
        # chart 데이터 → JSON 변환
        stocks_json = chart_data.to_json(orient="records", indent=4, force_ascii=False)

        indicators_json = df_indicators.to_json(
            orient="records", indent=4, force_ascii=False
        )
        # chart 데이터 → JSON 변환
        currency = "₩" if market == "dom" else "$"

        prompt_tech_user = f"""
        최근 일봉 차트 데이터({currency}): {stocks_json}
        기술적 지표: {indicators_json}
        """

        tech_report_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "technical_report",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intro": {"type": "string"},
                        "technical_report": {
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
                    "required": ["intro", "technical_report"],
                    "additionalProperties": False,
                },
            },
        }

        # 5. AI 분석 호출
        response = openai_client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "developer", "content": prompt_tech_dev},
                {"role": "user", "content": prompt_tech_user},
            ],
            reasoning_effort="low",
            response_format=tech_report_format,
        )

        tech_report = json.loads(response.choices[0].message.content)

        # 6. Markdown 리포트 생성
        report_md = f"{tech_report['intro']}\n\n"
        report_md += "---\n\n"
        for item in tech_report["technical_report"]:
            report_md += f"### {item['heading']}\n\n"
            content = item["content"].replace("~", "\\~")
            report_md += f"{content}\n\n"

        # 7. 리포트 저장
        report_path = self.report_manager.save_report(market, stock_code, report_md)

        print(
            f"💾 기술적 분석 리포트가 생성되었습니다.\n📁 저장 경로: {report_path}\n\n"
        )

        return report_md

    def get_technical_indicators(self, df, freq="daily") -> pd.DataFrame:
        """
        기술적 지표 계산
        freq의 기본 값을 daily로 설정.(app.py에 파라미터를 설정안해도 됨.)
        추후 확장성을 고려하여 주봉, 분봉 선택할 수 있게 메서드 내에 파라미터 설정.
        """
        if "last" in df.columns: # ta 라이브러리 사용 위해 컬럼 이름 통일
            df.rename(columns={"last": "close"}, inplace=True)

        # 차트 주기에 따라 지표 계산 기간 다르게 설정
        if freq == "min":
            # 분봉 기준 지표 (단기 중심)
            df_sma = df.ta.sma(length=10)
            df_ema = df.ta.ema(length=10)
            df_rsi = df.ta.rsi(length=7)
        elif freq == "weekly":
            # 주봉 기준 지표 (장기 중심)
            df_sma = df.ta.sma(length=60)
            df_ema = df.ta.ema(length=60)
            df_rsi = df.ta.rsi(length=21)
        else:
            # 일봉 기준 기본 지표
            df_sma = df.ta.sma(length=20)
            df_ema = df.ta.ema(length=20)
            df_rsi = df.ta.rsi(length=14)

        df_boll = df.ta.bbands(length=20, std=2) # 볼린저 밴드

        # 지표 병합
        df_indicators = pd.concat([df, df_sma, df_ema, df_rsi, df_boll], axis=1)

        return df_indicators


import asyncio  # 추가해주고

if __name__ == "__main__":
    import asyncio

    # 보고서 생성 테스트
    technical_analysis = TechnicalAnalysis()
    market = "ovs"
    ticker = "TSLA"
    report = asyncio.run(technical_analysis.get_report(market, ticker))
    print(report)