import json
import sys
import urllib.parse
import requests
import feedparser
import asyncio
from pathlib import Path

# project 루트 디렉토리를 파이썬 path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config_loader import openai_client  # ✅ config에서 불러옴

from data_loader import NewsFetcher

# ReportManager 가져오기
from analysis.report_manager import ReportManager


class NewsAnalysis:
    def __init__(self):
        # analysis_settings.json 파일 로드
        settings_path = Path(__file__).parent / "analysis_settings.json"
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # ReportManager 인스턴스 생성 (report_root, report_type, settings)
        self.report_manager = ReportManager(
            report_root=project_root / "report",
            report_type="news",  # 폴더 구분용
            settings=settings["news"],  # update_interval이 들어있는 딕셔너리
        )

        # 데이터 로더 인스턴스 생성
        self.news_fetcher = NewsFetcher()

    async def get_report(
        self, market: str, stock_code: str, overwrite=False, extra_prompt=""
    ) -> str:
        """
        뉴스 분석 보고서 생성
        :param market: 시장 구분
        :param stock_code: 종목 코드
        :param overwrite = False: 기존 리포트 덮어쓰기 여부
        """
        # ✅ 최신 리포트 파일 확인
        latest_report = self.report_manager.get_latest_report(market, stock_code)

        if latest_report:
            latest_time, latest_file = latest_report
            if not overwrite and not self.report_manager.is_update_required(
                latest_time
            ):
                return latest_file.read_text(encoding="utf-8")
            else:
                print(f"[업데이트 필요] {stock_code} 뉴스 보고서 갱신 중...")

        else:
            print(f"[신규 생성] {stock_code} 뉴스 보고서 생성 중...")

        # 뉴스 데이터 수집
        news_data = self.news_fetcher.get_news(market, stock_code)
        if news_data is None:
            return "뉴스 데이터를 찾을 수 없습니다."

        corp_name = self.news_fetcher.ticker_to_name(market, stock_code)

        # 2. 프롬프트 작성
        prompt_news_dev = f"""다음은 {corp_name}에 대한 최근 뉴스 기사입니다. 
제목을 살펴보고 감성 분석을 통해 {corp_name}의 향후 실적에 긍정적/부정적 영향을 미칠 것으로 예상되는 뉴스 기사를 4건 이상 선정하고 그 이유를 알려주세요.
같은 주제의 기사를 여러 개 골랐다면 그 중 하나만 선정해 주세요. 출처의 신뢰성과 중립성을 고려하여 선정하되, 긍정적, 부정적 소식을 골고루 검토해 주세요.

분석 결과는 주어진 JSON 스키마에 맞춰, 다음과 같은 내용을 포함합니다.

'index' 필드에는 주어진 뉴스 데이터에서 해당 기사의 인덱스를 반환합니다.
'title' 필드에는 기사의 제목을 반환합니다. 원본 제목을 그대로 사용해 주세요.
'reason' 필드에는 투자자의 관점에서 해당 기사를 중요하게 생각하는 이유를 구체적으로 설명합니다.
'sentiment' 필드에는 기사의 감성을 '긍정', '부정', '중립'으로 분류하고, 각각에 적절한 emoji와 함께 반환합니다.
'overall_sentiment' 필드에는 모든 기사의 감성을 평균내어 전체적인 감성을 반환합니다. 감성 점수는 -5 (매우 부정적: 대부분의 기사가 부정적)에서 5 (매우 긍정적: 대부분의 기사가 긍정적) 사이의 정수 값을 가집니다.
예를 들어, 총 4건의 기사를 선정했을 때 기사 2건을 '긍정', 1건을 '중립', 다른 1건을 '부정'으로 분류했다고 합시다. 이 때 전체 평가 점수는 (5 + 5 + 0 - 5) / 4 = 1.25, 반올림하여 2로 표시합니다.
'sentiment_emoji' 필드에는 전체적인 감성을 '🟢', '🔴', '🟡'으로 표시합니다. 각각은 긍정, 부정, 중립을 나타냅니다.

모든 내용은 한국어로 작성해 주세요.
        
        {extra_prompt}
        """

        # 기사 개수 제한(토큰 최대 방지를 위한 최근 100개까지만)
        MAX_ARTICLES = 100
        if len(news_data) > MAX_ARTICLES:
            # 뒤에서부터 잘라오기 → 최신 순
            news_data = news_data.tail(MAX_ARTICLES)

        news_json = news_data.to_json(orient="records", indent=4, force_ascii=False)

        prompt_news_user = f"뉴스 기사 목록(날짜, 제목, 출처, 링크): {news_json}"

        news_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "news_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "articles": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": {"type": "integer"},
                                    "title": {"type": "string"},
                                    "reason": {"type": "string"},
                                    "sentiment": {
                                        "type": "string",
                                        "enum": ["긍정 🌞", "부정 ⛈️", "중립 ⛅"],
                                    },
                                },
                                "required": ["index", "title", "reason", "sentiment"],
                                "additionalProperties": False,
                            },
                        },
                        "overall_sentiment": {"type": "integer"},
                        "sentiment_emoji": {
                            "type": "string",
                            "enum": ["🟢", "🔴", "🟡"],
                        },
                    },
                    "required": ["articles", "overall_sentiment", "sentiment_emoji"],
                    "additionalProperties": False,
                },
            },
        }

        # AI 분석 호출
        response = openai_client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "developer", "content": prompt_news_dev},
                {"role": "user", "content": prompt_news_user},
            ],
            reasoning_effort="medium",
            response_format=news_format,
        )

        try:
            analysis = json.loads(response.choices[0].message.content) # -> content None일 때 에러
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print("Problematic content:")
            print(response.choices[0].message.content)
            raise

        if not isinstance(analysis["articles"], list):
            article_list = [analysis["articles"]]
        else:
            article_list = analysis["articles"]

        # 리포트 Markdown 생성
        report_md = (
            f"# {corp_name} 관련 주요 뉴스 {len(article_list)}개를 선정했습니다.\n\n"
        )
        overall_sentiment = (
            f'+{analysis["overall_sentiment"]}'
            if analysis["overall_sentiment"] >= 0
            else f'{analysis["overall_sentiment"]}'
        )
        report_md += (
            f"### 전체 뉴스 감성: {overall_sentiment} {analysis['sentiment_emoji']}\n\n"
        )
        report_md += "---\n\n"

        for article in article_list:
            article_title = article["title"]

            # 링크 추출 (기존 뉴스 데이터에서 title 매칭)
            article_link = "#"
            if article_title in news_data["title"].values:
                article_link = news_data[news_data["title"] == article_title][
                    "link"
                ].values[0]
            else:
                encoded_title = urllib.parse.quote(article_title)
                url = (
                    f"https://news.google.com/rss/search?q={encoded_title}&hl=en-US&gl=US&ceid=US:en"
                    if market == "ovs"
                    else f"https://news.google.com/rss/search?q={encoded_title}&hl=ko&gl=KR&ceid=KR:ko"
                )
                response = requests.get(url)
                feed = feedparser.parse(response.text)
                article_link = feed.entries[0].link if feed.entries else "#"

            report_md += f"### [{article_title}]({article_link})\n"
            report_md += f"**감성**: {article['sentiment']}\n\n"
            report_md += f"**선정 이유**: {article['reason']}\n\n"
            report_md += "---\n\n"

        # 리포트 저장
        report_path = self.report_manager.save_report(market, stock_code, report_md)
        print(f"💾 뉴스 보고서가 생성되었습니다. \n📁 저장 경로: {report_path}\n\n")
        return report_md


if __name__ == "__main__":
    news_analysis = NewsAnalysis()

    asyncio.run(news_analysis.get_report("ovs", "MRNA", overwrite=True))