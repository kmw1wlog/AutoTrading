import sys
import pandas as pd
import requests
import pprint
import json
from pathlib import Path
from typing import Literal
from datetime import datetime, timedelta
from sec_cik_mapper import StockMapper

import requests
import zipfile
import io
import xml.etree.ElementTree as ET


# 프로젝트 루트 기준 경로
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config.config_loader import DART_API_KEY  # config에서 불러옴


class FinancialsFetcher:
    """
    재무제표 데이터를 관리하는 클래스
    - 국내(dom) 및 해외(ovs) 시장 데이터 지원
    - 데이터 저장, 병합, 필터링, API 호출 관리
    """

    def __init__(self):
        # 데이터 저장 경로
        self.data_financials_path = project_root / "data" / "financials"
        self._init_data_directories()
        self.update_year_threshold = (
            3  # 설정값을 config_loader에서 불러오면 여기 반영 가능
        )
        self.stock_mapper = StockMapper()

    def _init_data_directories(self):
        """데이터 저장을 위한 디렉토리 구조 초기화"""
        for market in ["dom", "ovs"]:
            market_path = self.data_financials_path / market
            market_path.mkdir(parents=True, exist_ok=True)

        # 국내 기업 info 폴더
        dom_info_path = self.data_financials_path / "dom" / "info"
        dom_info_path.mkdir(parents=True, exist_ok=True)

    def _save_financials_data(self, df: pd.DataFrame, market: str, ticker: str):
        """재무 데이터 저장"""
        save_path = self.data_financials_path / market / f"{ticker}.csv"
        df.to_csv(save_path, index=False, encoding="utf-8")
        print(
            f"[INFO] {market.upper()} {ticker} 재무 데이터 저장 완료 → {save_path.name}"
        )

    def _read_financials_data(self, market: str, ticker: str) -> pd.DataFrame | None:
        """기존 재무 데이터 로드"""
        file_path = self.data_financials_path / market / f"{ticker}.csv"

        # 파일이 없으면 바로 None 반환 → 조회 필요
        if not file_path.exists():
            print(
                f"[INFO] {market.upper()} {ticker} 재무 데이터 파일 없음 → 신규 조회 필요"
            )
            return None

        try:
            df = pd.read_csv(file_path)

            # 데이터가 비어있으면 None 반환
            if df.empty:
                print(
                    f"[WARN] {market.upper()} {ticker} 재무 데이터가 비어있음 → 신규 조회 필요"
                )
                return None

            print(f"[INFO] {market.upper()} {ticker} 기존 재무 데이터 로드 성공")
            return df

        except pd.errors.EmptyDataError:
            # 파일은 있는데 내용이 없거나 잘못된 경우
            print(
                f"[ERROR] {market.upper()} {ticker} CSV 파일이 비어있거나 손상됨 → 신규 조회 필요"
            )
            return None

        except Exception as e:
            print(f"[ERROR] {market.upper()} {ticker} 재무 데이터 로드 실패: {e}")
            return None

    def get_financials(self, market: Literal["dom", "ovs"], stock_code: str):
        print(f"[INFO] {market.upper()} {stock_code} 재무 데이터 조회 시작")

        # 기존 데이터 읽기
        existing_data = self._read_financials_data(market, stock_code)
        # print(" existing_data: ", existing_data)
        # 기존 데이터가 있다면 최신성 확인
        if existing_data is not None:
            latest_year = self._get_latest_year(
                existing_data, market
            )  # 둘 다 이 함수로 처리한다
            this_year = datetime.now().year

            print(
                f"[DEBUG] {market.upper()} {stock_code} → latest_year: {latest_year}, this_year: {this_year}"
            )

            if (this_year - latest_year) < self.update_year_threshold:
                print(f"[INFO] {stock_code} 데이터 최신 → 기존 데이터 반환")
                return existing_data

            print(f"[INFO] {stock_code} 데이터 갱신 필요 → 업데이트 진행")
        else:
            print(f"[INFO] {stock_code} 기존 재무 데이터 없음 → 최초 수집 시작")

        # 데이터가 없거나 업데이트가 필요하면 새로 수집
        if market == "dom":
            print(f"[WARN] {market.upper()} {stock_code} 재무 데이터 조회 실패")
            df = self._fetch_domestic_financials(stock_code)
        else:
            df = self._fetch_overseas_financials(stock_code)

        # 수집 실패 시 None 반환
        if df.empty:
            print(f"[WARN] {market.upper()} {stock_code} 재무 데이터 조회 실패")
            return None

        # 수집 성공 시 저장 후 반환
        self._save_financials_data(df, market, stock_code)
        return df

    def _get_latest_year(self, df: pd.DataFrame, market: str) -> int:
        try:
            if market == "dom":
                if "bsns_year" in df.columns:
                    years = pd.to_numeric(df["bsns_year"], errors="coerce").dropna()
                    if years.empty:
                        raise ValueError("bsns_year 컬럼이 비어 있음.")
                elif "thstrm_nm" in df.columns:
                    years = (
                        df["thstrm_nm"]
                        .str.extract(r"제\s*(\d+)\s*기")[0]
                        .astype(float)
                        .dropna()
                    )
                    if years.empty:
                        raise ValueError("thstrm_nm 컬럼 파싱 실패.")
                else:
                    raise ValueError("유효한 연도 컬럼 없음.")
            else:  # "ovs"
                if "Date" not in df.columns:
                    raise ValueError("해외 데이터에 Date 컬럼이 없음.")
                dates = pd.to_datetime(df["Date"], errors="coerce")
                years = dates.dt.year.dropna()

            latest_year = int(years.max())
            print(f"[DEBUG] {market.upper()} 최근 연도: {latest_year}")
            return latest_year

        except Exception as e:
            print(f"[WARN] 최근 연도 파악 실패: {e}")
            return 0

    def _fetch_overseas_financials(self, ticker: str) -> pd.DataFrame:
        """SEC에서 해외 기업 재무 데이터 조회"""
        print(f"[INFO] 해외 기업 {ticker} 재무 데이터 수집 시작")
        HEADERS = {
            "User-Agent": "example@example.com",  # 사용자 정보 기입 필요
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
            "Connection": "keep-alive",
        }

        cik = self.stock_mapper.ticker_to_cik[ticker]
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        response = requests.get(url, headers=HEADERS)

        if response.status_code != 200:
            print(f"[ERROR] SEC 데이터 요청 실패: {response.status_code}")
            return pd.DataFrame()

        data = response.json()
        us_gaap = data.get("facts", {}).get("us-gaap", {})

        financial_data = []
        existing_entries = set()
        three_years_ago = datetime.now() - timedelta(days=3 * 365)
        # metric (재무 항목명, 예: Revenues, NetIncomeLoss)
        for metric, details in us_gaap.items():
            for unit, entries in details["units"].items():
                sorted_entries = sorted(
                    entries, key=lambda e: (e.get("end", ""), e.get("filed", ""))
                )

                for entry in sorted_entries:
                    # 날짜 정보 가져오기
                    end_date = entry.get("end", "")
                    # 회계 연말 데이터만 선택
                    if entry.get("fp", "") != "FY":
                        continue

                    # 최근 3년치만 활용
                    formatted_end_date = datetime.strptime(end_date, "%Y-%m-%d")
                    if formatted_end_date <= three_years_ago:
                        continue

                    # 중복 데이터 처리: 최신 데이터로 덮어쓰기
                    key = (end_date, metric)
                    if key in existing_entries:
                        financial_data.pop()
                    existing_entries.add(key)

                    financial_data.append(
                        {
                            "Date": end_date,
                            "Metric": metric,
                            "Unit": unit,
                            "Value": entry.get("val", ""),
                        }
                    )

        df = pd.DataFrame(financial_data)
        if not df.empty:
            print(f"[INFO] 해외 기업 {ticker} 재무 데이터 수집 완료")
        return df

    def _fetch_domestic_financials(self, ticker: str) -> pd.DataFrame:
        """DART에서 국내 기업 재무 데이터 조회"""
        print(f"[INFO] 국내 기업 {ticker} 재무 데이터 수집 시작")

        corp_code, stock_code, corp_name = self._load_dom_corp_info(ticker)
        end_year = datetime.now().year - 1

        url = (
            f"https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json?"
            f"crtfc_key={DART_API_KEY}&corp_code={corp_code}&bsns_year={end_year}&reprt_code=11011&fs_div=OFS"
        )

        response = requests.get(url)
        data = response.json()
        print(
            f"[DEBUG] message: {data.get('message', '없음')}, 최초 조회 일자: {data.get('list', [{}])[0].get('rcept_no', '없음')}"
        )

        if "list" not in data:
            print(
                f"[ERROR] {corp_name}({stock_code}) 재무 데이터 조회 실패: {data.get('message', '알 수 없는 오류')}"
            )
            return pd.DataFrame()

        df = pd.DataFrame(data["list"])

        # bsns_year는 유지한다!
        df = df.drop(
            columns=[
                "rcept_no",
                "reprt_code",
                "corp_code",
                "sj_div",
                "ord",
                "thstrm_add_amount",
            ]
        )

        print(f"[INFO] 국내 기업 {corp_name}({stock_code}) 재무 데이터 수집 완료")

        return df

    # XML파일 저장 및 조회
    def _fetch_dom_corp_info(self):
        """DART에서 전체 국내 기업 기본 정보 수집"""
        url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
        response = requests.get(url)

        corp_data = []
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            with zip_file.open("CORPCODE.xml") as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for child in root.findall("list"):
                    corp_code = child.find("corp_code").text
                    stock_code = child.find("stock_code").text
                    corp_name = child.find("corp_name").text

                    if stock_code.strip():
                        corp_data.append(
                            {
                                "corp_code": corp_code,
                                "stock_code": stock_code,
                                "corp_name": corp_name,
                            }
                        )

        df = pd.DataFrame(corp_data)
        save_path = self.data_financials_path / "dom" / "info" / "corp_info.csv"
        df.to_csv(save_path, index=False)
        print(f"[INFO] 전체 기업 정보 저장 완료 → {save_path.name}")

        return df

    def _load_dom_corp_info(self, ticker: str):
        """corp_info.csv 에서 ticker 에 해당하는 기업 코드와 이름 반환"""
        save_path = self.data_financials_path / "dom" / "info" / "corp_info.csv"

        # 1. 파일 존재 여부 먼저 검사
        if not save_path.exists():
            print(f"[INFO] corp_info.csv 파일 없음 → 새로 생성합니다.")
            df = self._fetch_dom_corp_info()
        else:
            df = pd.read_csv(save_path, dtype=str)

        result = df[df["stock_code"] == ticker]

        # 2. ticker 가 없으면 DART 재조회
        if result.empty:
            print(
                f"[WARN] '{ticker}' 종목이 corp_info.csv 에 없음 → DART 재조회합니다."
            )
            df = self._fetch_dom_corp_info()
            result = df[df["stock_code"] == ticker]

        corp_code = result.iloc[0]["corp_code"]
        corp_name = result.iloc[0]["corp_name"]

        return corp_code, ticker, corp_name

    def ticker_to_company_name(
        self, market: Literal["dom", "ovs"], stock_code: str
    ) -> str:
        """종목 코드로 기업명 반환"""
        try:
            if market == "dom":
                _, _, corp_name = self._load_dom_corp_info(stock_code)
            else:
                corp_name = self.stock_mapper.ticker_to_company_name[stock_code]
            return corp_name
        except Exception:
            raise ValueError(f"[ERROR] '{stock_code}'는 유효한 종목 코드가 아닙니다.")


if __name__ == "__main__":
    fetcher = FinancialsFetcher()

    # 테스트 실행
    temp = fetcher._fetch_dom_corp_info()
    print(temp)