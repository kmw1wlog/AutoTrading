from pathlib import Path
import sys
import json
import feedparser
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Literal

# project 루트 디렉토리를 파이썬 path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_loader.financials import FinancialsFetcher
import urllib.parse

class NewsFetcher:
    """
    뉴스 데이터를 관리하는 클래스
    - 국내(dom) 및 해외(ovs) 시장 뉴스 데이터 지원
    - 데이터 저장, 병합, 필터링, API 호출 관리
    """

    def __init__(self):
        # settings.json 파일 로드
        settings_path = Path(__file__).parent / "settings.json"
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # 뉴스 업데이트 기준 시간 설정(ex: 24시간)
        self.update_threshold = timedelta(
            hours=settings["news_collection"]["hours_threshold"]
        )

        # 데이터 저장 경로 설정
        self.data_news_path = project_root / "data" / "news"
        self._init_data_directories()

        # 종목코드-기업명 매핑을 위한 객체들 초기화
        self.ticker_to_name = FinancialsFetcher().ticker_to_company_name

    def _init_data_directories(self):
        """데이터 저장을 위한 디렉토리 구조 초기화"""
        # dom/ovs 디렉토리
        for market in ["dom", "ovs"]:
            market_path = self.data_news_path / market
            market_path.mkdir(parents=True, exist_ok=True)

    def get_news(
        self,
        market: Literal["dom", "ovs"],
        stock_code: str,
    ) -> pd.DataFrame | None:
        """현재로부터 최신 뉴스 데이터를 조회하고 저장/반환하는 메서드
        Args:
            market: 시장 구분 (dom: 국내, ovs: 해외)
            stock_code: 종목 코드

        Returns:
            pd.DataFrame | None: 뉴스 데이터 (datetime 인덱스)
            조회 실패 시 None 반환

        Process:
            1. 기존 뉴스 데이터가 있는 경우, 최신 여부 판단
            2. 최신이면 그대로 반환, 오래됐으면 새로운 뉴스 수집
            3. 기존 뉴스와 새 뉴스를 병합 및 정렬
            4. 병합된 데이터를 저장하고 반환
        """
        now = datetime.now()

        # 기존 데이터 로드
        existing_data = self._read_news_data(market, stock_code)

        if existing_data is not None and not existing_data.empty:
            last_update_time = existing_data.index[-1]

            # 업데이트 기준 시간 이내면 기존 데이터 반환
            if now - last_update_time < self.update_threshold:
                print("[INFO] 업데이트 기준시간 내 → 기존 데이터 반환")
                return existing_data

            combined_data = existing_data.copy()

        else:
            print("[INFO] 기존 뉴스 데이터 없음 → 최초 수집 시작")
            combined_data = pd.DataFrame()

        # 새로운 뉴스 데이터 수집
        fetched_data = self._fetch_news(market, stock_code)

        if fetched_data.empty:
            print(f"[WARN] 새로운 뉴스 데이터 없음 → 반환할 데이터가 없습니다.")
            return combined_data if not combined_data.empty else None
        else:
            print(f"[INFO] 기업 {stock_code} 재무 데이터 업데이트 완료")
        # 기존 데이터 병합 및 정렬
        combined_data = pd.concat([combined_data, fetched_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
        combined_data.sort_index(inplace=True)

        # 최종 데이터 저장 + 반환
        final_data = self._save_news_data(combined_data, market, stock_code)

        return final_data

    def _fetch_news(
        self, market: Literal["dom", "ovs"], stock_code: str
    ) -> pd.DataFrame:
        """Google News RSS에서 최신 뉴스 가져오기

        Args:
            stock_code (str): 종목 코드

        Returns:
            pd.DataFrame: 뉴스 데이터 (datetime 인덱스)
        """
        stock_name = self.ticker_to_name(market, stock_code)
        encoded_stock_name = urllib.parse.quote(stock_name)  # URL 인코딩

        # 시장 구분(국내/해외)에 따라 언어 설정
        if market == "dom":
            url = f"https://news.google.com/rss/search?q={encoded_stock_name}&hl=ko&gl=KR&ceid=KR:ko"
        else:
            url = f"https://news.google.com/rss/search?q={encoded_stock_name}&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url)
        feed = feedparser.parse(response.text)
        news_list = []
        for entry in feed.entries:
            news_list.append(
                {
                    "datetime": datetime.strptime(
                        entry.published, "%a, %d %b %Y %H:%M:%S %Z"
                    ),
                    "title": entry.title,
                    "source": (
                        entry.source.title if hasattr(entry, "source") else "Unknown"
                    ),
                    "link": entry.link,
                }
            )

        df = pd.DataFrame(news_list)

        if not df.empty:
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)  # 시간 오름차순 정렬

        return df

    def _read_news_data(self, market: str, stock_code: str) -> pd.DataFrame | None:
        """
        기존 뉴스 데이터 파일을 읽어오기
        """
        save_dir = self.data_news_path / market
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / f"{stock_code}.csv"

        if not file_path.exists():
            return None

        if file_path.stat().st_size == 0:
            print(f"[WARN] 파일이 비어 있습니다: {file_path.name}")
            return None

        df = pd.read_csv(file_path, index_col="datetime", parse_dates=True)

        if df.empty:
            print(f"[WARN] 파일 읽었지만 데이터가 없습니다: {file_path.name}")
            return None

        return df

    def _save_news_data(
        self, new_df: pd.DataFrame, market: str, stock_code: str
    ) -> pd.DataFrame:
        """
        뉴스 데이터 저장 메서드
        """
        save_dir = self.data_news_path / market
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / f"{stock_code}.csv"

        # 기존 데이터와 병합은 get_news에서 이미 처리함
        # 여기서는 병합된 최종 데이터 저장만 수행
        filtered_df = self._filter_incomplete_news(new_df)

        if filtered_df.empty:
            print(f"[WARN] 저장할 뉴스 데이터가 없습니다: {stock_code}")
            return filtered_df

        filtered_df.to_csv(file_path)

        return filtered_df

    def _filter_incomplete_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        뉴스 데이터에서 안전하게 완성된 뉴스만 반환
        settings.json의 hours_threshold 기준 사용
        """
        now = datetime.now()

        # news_collection 기준으로 안전 cutoff 생성
        safe_cutoff = now - self.update_threshold

        # 기준 시간 이전 뉴스만 유지
        return df[df.index <= safe_cutoff]


if __name__ == "__main__":
    # 뉴스 가져오기 테스트
    news_fetcher = NewsFetcher()
    market = "ovs"
    stock_code = "MRNA"
    news = news_fetcher.get_news(market, stock_code)
    print(news)
