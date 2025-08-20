# 한국투자증권 API 사용하는 모듈 - 직접 소통은 모두 여기서
import json
import time
import requests
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta
from sec_cik_mapper import StockMapper

# ✅ config_loader 불러오기
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from config.config_loader import config

class Kis:
    def __init__(self, token_path: Path = None):
        # ✅ config_loader에서 kis 섹션 바로 불러오기
        self.config = config["kis"]  # kis 관련 설정만 추출!

        self.base_url = self.config["url_base"]
        self.app_key = self.config["app_key"]
        self.app_secret = self.config["app_secret"]
        self.token_path = token_path or (Path(__file__).parent / "token.json")

        # 접근 토큰 발급
        self.access_token = self._get_access_token()

        # 시장코드 동적 할당을 위한 초기화
        self.ticker_to_exchange = StockMapper().ticker_to_exchange
        self.exchange_codes = {
            "Nasdaq": "NAS",
            "NYSE": "NYS",
            "OTC": None,
            "CBOE": None,
        }

    def _get_access_token(self) -> str | None:
        """한국투자증권 API 접근 토큰을 세팅하는 메서드
        유효기간이 남은 기존 토큰을 사용하거나, 새로운 토큰을 발급받아 세팅한다.

        Returns:
            str | None: 유효한 접근 토큰 또는 발급 실패시 None

        Process:
            1. token.json 파일 존재하면 기존 토큰 유효성 검사
            2. 유효한 토큰 있으면 재사용
            3. 없거나 만료되었으면 새로운 토큰 발급 요청
            4. 발급받은 토큰을 파일에 저장하고 반환
        """
        token_file = self.token_path  # 여기만 바뀜

        # 기존 토큰 파일이 있는지 확인하고 읽기
        if token_file.exists():
            with open(token_file, "r", encoding="utf-8") as f:
                token_data = json.load(f)
                token = token_data.get("access_token")
                issued_time = token_data.get("issued_time")

                # 토큰 유효성 검사 (24시간 이내 발급)
                if token and issued_time:
                    issued_time = datetime.fromisoformat(issued_time)
                    if datetime.now() - issued_time < timedelta(hours=24):
                        return token

        # 토큰이 없거나 만료된 경우 새로운 토큰 발급 요청
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        path = "oauth2/tokenP"
        url = f"{self.base_url}/{path}"

        # API 호출 제한 방지를 위한 지연
        time.sleep(0.1)
        res = requests.post(url, headers=headers, data=json.dumps(body))

        if res.status_code == 200:
            try:
                new_token = res.json().get("access_token")
                # 새로운 토큰과 발급 시간을 파일에 저장
                with open(token_file, "w", encoding="utf-8") as f:
                    json.dump({"access_token": new_token,"issued_time": datetime.now().isoformat()}, f)
                print(f"새로운 접근 토큰 발급: {new_token[:10]}...")
                print(f"토큰 발급 시각: {datetime.now().isoformat()}")
                return new_token
            except KeyError as e:
                print(f"접근 토큰 발급 중 오류 발생: {e}")
                print(res.json())
                return None
        else:
            print("접근 토큰 발급 실패. 응답 코드:", res.status_code)
            print("응답 내용:", res.json())
            return None

    def resolve_overseas_market_code(self, ticker: str) -> str | None:
        """
        종목 코드(ticker)로부터 KIS API용 해외 마켓 코드(NAS, NYS 등)를 반환

        Args:
            ticker (str): 해외 종목 코드 (예: 'AAPL', 'XOM')

        Returns:
            str | None: 시장 코드 (예: 'NAS', 'NYS'), 못 찾으면 None
        """
        exchange = self.ticker_to_exchange.get(ticker)
        if exchange is None:
            raise ValueError(f"[ERROR] 거래소 정보가 없습니다: {ticker}")
        market_code = self.exchange_codes.get(exchange)
        if market_code is None:
            raise ValueError(f"[ERROR] 지원하지 않는 거래소입니다: {exchange}")
        return market_code

    def get_minute_chart_today(
        self, stock_code: str, time_from: str = "105000"
    ) -> pd.DataFrame:  # 국내 당일 분봉
        """특정 종목의 당일 분봉 데이터를 조회하는 메서드

        Args:
            stock_code (str): 종목 코드 (예: '005930')
            time_from (str): 조회 시작 시각 (HHMMSS 형식, 기본값: '105000')

        Returns:
            pd.DataFrame: 분봉 데이터 (columns: [open, high, low, close, volume], index: datetime)
                - datetime: 체결 시각 (datetime)
                - open: 시가
                - high: 고가
                - low: 저가
                - close: 종가
                - volume: 거래량

        Process:
            1. API 엔드포인트 URL과 인증 헤더 설정
            2. 분봉 조회 파라미터 설정 후 API 요청
            3. 응답 데이터를 DataFrame으로 변환
        """
        # API 엔드포인트 설정
        path = "/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        url = f"{self.base_url}/{path}"

        # API 요청 헤더 설정 (인증 정보 포함)
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST03010200",  # 국내 주식 당일 분봉 조회용 TR_ID
            "custtype": self.config["customer_type"],
        }

        # 분봉 조회 파라미터 설정
        params = {
            "fid_etc_cls_code": "",
            "fid_cond_mrkt_div_code": "J",  # 시장구분 - 주식
            "fid_input_iscd": stock_code,  # 종목코드
            "fid_input_hour_1": time_from,  # 조회 시작시각
            "fid_pw_data_incu_yn": "N",  # 수정주가 여부
        }

        time.sleep(0.1)  # API 호출 제한 방지를 위한 지연
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        # output2 데이터를 DataFrame으로 변환
        df = pd.DataFrame(data["output2"])

        # 날짜와 시간 컬럼 결합하여 datetime 생성
        df["datetime"] = pd.to_datetime(
            df["stck_bsop_date"] + df["stck_cntg_hour"], format="%Y%m%d%H%M%S"
        )

        # 필요한 컬럼만 선택하고 이름 변경
        df = df.rename(
            columns={
                "stck_oprc": "open",
                "stck_hgpr": "high",
                "stck_lwpr": "low",
                "stck_prpr": "close",
                "cntg_vol": "volume",
            }
        )[["datetime", "open", "high", "low", "close", "volume"]]

        # 데이터 타입 변환
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        # datetime 기준 오름차순 정렬 및 인덱스 설정
        df = df.sort_values("datetime").set_index("datetime")

        return df

    def get_minute_chart_past(
        self, stock_code: str, date: str, time_from: str = "140000"
    ) -> pd.DataFrame:  # 국내 과거 분봉
        """특정 종목의 과거 분봉 데이터를 조회하는 메서드

        Args:
            stock_code (str): 종목 코드 (예: '005930')
            date (str): 조회 날짜 (YYYYMMDD 형식, 예: '20241108')
            time_from (str): 조회 시작 시각 (HHMMSS 형식, 기본값: '140000')

        Returns:
            pd.DataFrame: 분봉 데이터 (columns: [open, high, low, close, volume], index: datetime)
                - datetime: 체결 시각 (datetime)
                - open: 시가
                - high: 고가
                - low: 저가
                - close: 종가
                - volume: 거래량
            (조회 시각 - 119분) ~ (조회 시각): 총120개 데이터 조회

        Process:
            1. API 엔드포인트 URL과 인증 헤더 설정
            2. 과거 분봉 조회 파라미터 설정 후 API 요청
            3. 응답 데이터를 DataFrame으로 변환하고 datetime 컬럼 생성 후 정렬
        """
        # API 엔드포인트 설정
        path = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
        url = f"{self.base_url}/{path}"

        # API 요청 헤더 설정 (인증 정보 포함)
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST03010230",  # 국내 주식 일별 분봉 조회용 TR_ID
            "custtype": self.config["customer_type"],
        }

        # 과거 분봉 조회 파라미터 설정
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 시장구분 - 주식
            "FID_INPUT_ISCD": stock_code,  # 종목코드
            "FID_INPUT_DATE_1": date,  # 조회 날짜 (YYYYMMDD)
            "FID_INPUT_HOUR_1": time_from,  # 조회 시작시각 (HHMMSS)
            "FID_PW_DATA_INCU_YN": "Y",  # 과거 데이터 포함 여부
            "FID_FAKE_TICK_INCU_YN": "N",  # 허봉 포함 여부
        }

        time.sleep(0.1)  # API 호출 제한 방지를 위한 지연
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        # output2 데이터를 DataFrame으로 변환
        df = pd.DataFrame(data["output2"])

        # 날짜와 시간 컬럼을 결합하여 datetime 생성
        df["datetime"] = pd.to_datetime(
            df["stck_bsop_date"] + df["stck_cntg_hour"], format="%Y%m%d%H%M%S"
        )

        # 필요한 컬럼만 선택하고 이름 변경
        df = df.rename(
            columns={
                "stck_oprc": "open",
                "stck_hgpr": "high",
                "stck_lwpr": "low",
                "stck_prpr": "close",
                "cntg_vol": "volume",
            }
        )[["datetime", "open", "high", "low", "close", "volume"]]

        # 데이터 타입 변환
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        # datetime 기준 오름차순 정렬 및 인덱스 설정
        df = df.sort_values("datetime").set_index("datetime")

        return df

    def get_period_chart(
        self, stock_code: str, start_date: str, end_date: str, period_code: str = "D"
    ) -> pd.DataFrame:  # 국내 일/주/월/년
        """특정 종목의 기간별 주가 데이터를 조회하는 메서드

        Args:
            stock_code (str): 종목 코드 (예: '005930')
            start_date (str): 조회 시작일자 (YYYYMMDD 형식)
            end_date (str): 조회 종료일자 (YYYYMMDD 형식)
            period_code (str): 조회 주기 구분 코드 (기본값: 'D' - 일봉; 'W' - 주봉)

        Returns:
            pd.DataFrame: 일봉 데이터 (columns: [open, high, low, close, volume], index: datetime)
                - datetime: 거래일자 (datetime)
                - open: 시가
                - high: 고가
                - low: 저가
                - close: 종가
                - volume: 거래량

        Process:
            1. API 엔드포인트 URL과 인증 헤더 설정
            2. 일봉 조회 파라미터 설정 후 API 요청
            3. 응답 데이터를 DataFrame으로 변환
        """
        # API 엔드포인트 설정
        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        url = f"{self.base_url}/{path}"

        # API 요청 헤더 설정 (인증 정보 포함)
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST03010100",  # 국내 주식 일봉(기간별 시세) 조회용 TR_ID
            "custtype": self.config["customer_type"],
        }

        # 일봉 조회 파라미터 설정
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # 시장구분 - 주식
            "FID_INPUT_ISCD": stock_code,  # 종목코드
            "FID_INPUT_DATE_1": start_date,  # 조회 시작일자
            "FID_INPUT_DATE_2": end_date,  # 조회 종료일자
            "FID_PERIOD_DIV_CODE": period_code,  # 조회 주기 구분코드
            "FID_ORG_ADJ_PRC": "0",  # 수정주가 여부 (0: 수정주가, 1: 원가)
        }

        time.sleep(1)  # API 호출 제한 방지를 위한 지연
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        # 추가!
        # print("응답 데이터 확인 >>>", json.dumps(data, indent=2, ensure_ascii=False))

        # output2 데이터를 DataFrame으로 변환
        df = pd.DataFrame(data["output2"])

        # 날짜 컬럼을 datetime으로 변환
        try:
            df["datetime"] = pd.to_datetime(df["stck_bsop_date"], format="%Y%m%d")
        except KeyError:  # Empty response due to short time frame since last update
            return pd.DataFrame()  # empty dataframe

        # 필요한 컬럼만 선택하고 이름 변경
        df = df.rename(
            columns={
                "stck_oprc": "open",
                "stck_hgpr": "high",
                "stck_lwpr": "low",
                "stck_clpr": "close",
                "acml_vol": "volume",
            }
        )[["datetime", "open", "high", "low", "close", "volume"]]

        # 데이터 타입 변환
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        # datetime 기준 오름차순 정렬 및 인덱스 설정
        df = df.sort_values("datetime").set_index("datetime")

        return df

    def get_dom_detail(self, stock_code: str) -> dict:
        """국내 주식의 상세 정보를 조회하여
        주요 지표(PER, PBR, EPS, BPS)만 저장 후 반환
        (참조. https://apiportal.koreainvestment.com/apiservice-apiservice?/uapi/domestic-stock/v1/quotations/inquire-price)
        """
        path = "/uapi/domestic-stock/v1/quotations/inquire-price"
        url = f"{self.base_url}/{path}"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST01010100",  # 국내 주식 현재가 시세 조회용 TR_ID
        }
        params = {
            "fid_etc_cls_code": "",
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": stock_code,
        }
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        df = pd.DataFrame(data["output"], index=[0])
        return df

    def get_dom_financial_ratio(self, stock_code: str) -> pd.DataFrame:
        """
        국내 주식의 재무비율 정보를 조회하여
        주요 지표(ROE, EPS, BPS, stac_yymm)를 DataFrame으로 반환
        Args:
            stock_code (str): 종목 코드 (예: '005930')
        Returns:
            pd.DataFrame: 재무비율 데이터 (columns: [ROE, EPS, BPS, stac_yymm])
        """
        path = "/uapi/domestic-stock/v1/finance/financial-ratio"
        url = f"{self.base_url}/{path}"

        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST66430300",
            "custtype": self.config["customer_type"],
        }

        params = {
            "FID_DIV_CLS_CODE": "1",  # 0: 연간, 1: 분기
            "fid_cond_mrkt_div_code": "J",  # 주식시장
            "fid_input_iscd": stock_code,  # 종목코드
        }

        time.sleep(1)
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        if "output" not in data or not data["output"]:
            print("국내 데이터 조회 실패:", data)
            return pd.DataFrame()

        df = pd.DataFrame(data["output"])

        # stac_yymm + 주요 지표만 추출
        df = df[["stac_yymm", "roe_val", "eps", "bps"]]

        # 컬럼명 정리
        df.rename(
            columns={
                "stac_yymm": "stac_yymm",  # 결산년월 (그대로)
                "roe_val": "ROE",
                "eps": "EPS",
                "bps": "BPS",
            },
            inplace=True,
        )

        # 숫자 타입으로 변환 (혹시 몰라서)
        df["ROE"] = pd.to_numeric(df["ROE"], errors="coerce")
        df["EPS"] = pd.to_numeric(df["EPS"], errors="coerce")
        df["BPS"] = pd.to_numeric(df["BPS"], errors="coerce")

        # stac_yymm를 인덱스로 사용할 경우 (옵션)
        # df.set_index("stac_yymm", inplace=True)

        return df

    def get_overseas_period_chart(
        self,
        stock_code: str,
        end_date: str,
        period_code: int = 0,
    ) -> pd.DataFrame:  # 해외 일/주/월
        """해외 주식의 기간별 주가 데이터를 조회하는 메서드
        참조.

        Args:
            market_code (str): 시장 분류 코드 (예: 'NAS': 나스닥, 'NYS': 뉴욕, ...)
            stock_code (str): 종목 코드 (예: 'TSLA')
            end_date (str): 조회 종료일자 (YYYYMMDD 형식)
            period_code (str): 조회 주기 구분 코드 (기본값: '0' - 일봉, '1' - 주봉, '2' - 월봉)

        Returns:
            pd.DataFrame: 주가 데이터 (columns: [open, high, low, close, volume], index: datetime)
                - datetime: 거래일자 (datetime)
                - open: 시가
                - high: 고가
                - low: 저가
                - close: 종가
                - volume: 거래량

        Process:
            1. API 엔드포인트 URL과 인증 헤더 설정
            2. 기간별 시세 조회 파라미터 설정 후 API 요청
            3. 응답 데이터를 DataFrame으로 변환
        """
        # 시장 코드 조회
        market_code = self.resolve_overseas_market_code(stock_code)

        # API 엔드포인트 설정
        path = "/uapi/overseas-price/v1/quotations/dailyprice"
        url = f"{self.base_url}/{path}"


        # API 요청 헤더 설정 (인증 정보 포함)
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "HHDFS76240000",  # 해외 주식 일봉(기간별 시세) 조회용 TR_ID
            "custtype": self.config["customer_type"],
        }

        # 일봉 조회 파라미터 설정
        params = {
            "AUTH": "",  # 기본 값 (Null)
            "EXCD": market_code,  # 거래소 코드 (예: 'NAS', 'NYS')
            "SYMB": stock_code,  # 종목 코드 (예: 'TSLA')
            "GUBN": period_code,  # 조회 구분 (0: 일, 1: 주, 2: 월)
            "BYMD": end_date,  # 조회 기준 일자: 한 번 호출에 이 날까지 100건 조회
            "MODP": "1",  # 수정주가 반영 여부 (0: 미반영, 1: 반영)
        }

        time.sleep(1)  # API 호출 제한 방지를 위한 지연
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        # output2 데이터를 DataFrame으로 변환
        df = pd.DataFrame(data["output2"])

        try:
            df["datetime"] = pd.to_datetime(df["xymd"], format="%Y%m%d")
        except KeyError:  # 마지막 업데이트 이후 데이터 없음
            print(data["msg1"])
            return pd.DataFrame()  # 빈 DataFrame 반환

        # 필요한 컬럼만 선택하고 이름 변경
        df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "clos": "close",
                "tvol": "volume",
            },
            inplace=True,
        )

        df = df[["datetime", "open", "high", "low", "close", "volume"]]

        # 데이터 타입 변환
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        # datetime 기준 오름차순 정렬 및 인덱스 설정
        df = df.sort_values("datetime").set_index("datetime")

        return df

    def get_overseas_minute_chart(
        self,
        stock_code: str,
        interval: int = 1,
        include_prev_day: bool = False,
    ) -> pd.DataFrame:  # 해외 분봉
        """해외 주식의 분봉 데이터를 조회하는 메서드

        Args:
            market_code (str): 거래소 코드 (예: 'NAS' - 나스닥, 'NYS' - 뉴욕증권거래소 등)
            stock_code (str): 종목 코드 (예: 'TSLA')
            date (str): 조회 기준일자 (YYYYMMDD 형식, 예: '20240201')
            interval (int): 분봉 간격 (기본값: 1분봉, 예: 1, 5, 10)
            include_prev_day (bool): 전일 포함 여부 ( True: 포함, False: 미포함)

        Returns:
            pd.DataFrame: 분봉 데이터 (columns: [datetime, open, high, low, close, volume])
                - datetime: 체결 시각 (datetime)
                - open: 시가
                - high: 고가
                - low: 저가
                - close: 종가
                - volume: 거래량

        Process:
            1. API 엔드포인트 URL과 인증 헤더 설정
            2. 분봉 조회 파라미터 설정 후 API 요청
            3. 응답 데이터를 DataFrame으로 변환

        Memo:
            - 해외 주식 분봉의 경우 당일 기준으로만 조회가 가능합니다. (조회 기간 설정 불가)
        """
        # 시장 코드 조회
        market_code = self.resolve_overseas_market_code(stock_code)
        # API 엔드포인트 설정
        path = "/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"
        url = f"{self.base_url}/{path}"

        # API 요청 헤더 설정 (인증 정보 포함)
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "HHDFS76950200",  # 해외 주식 분봉 조회용 TR_ID
            "custtype": self.config["customer_type"],
        }

        # 과거 분봉 조회 파라미터 설정
        params = {
            "AUTH": "",
            "EXCD": market_code,  # 거래소 코드 (예: 'NAS', 'NYS')
            "SYMB": stock_code,  # 종목 코드 (예: 'TSLA')
            "NMIN": str(interval),  # 분봉 간격 (1: 1분봉, 2: 2분봉, ...)
            "PINC": "1" if include_prev_day else "0",  # 전일 포함 여부
            "NEXT": "",  # 처음 조회 시 공백
            "NREC": "120",  # 최대 120건 요청
            "FILL": "",
            "KEYB": "",  # 처음 조회 시 공백
        }

        time.sleep(0.1)  # API 호출 제한 방지를 위한 지연
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        # output2 데이터를 DataFrame으로 변환
        df = pd.DataFrame(data["output2"])

        # 날짜와 시간 컬럼을 결합하여 datetime 생성
        df["datetime"] = pd.to_datetime(df["xymd"] + df["xhms"], format="%Y%m%d%H%M%S")

        # 필요한 컬럼만 선택하고 이름 변경
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "last": "close",
                "evol": "volume",
            }
        )[["datetime", "open", "high", "low", "close", "volume"]]

        # 데이터 타입 변환
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

        # datetime 기준 오름차순 정렬 및 인덱스 설정
        df = df.sort_values("datetime").set_index("datetime")

        return df

    def get_overseas_detail(self, stock_code: str) -> dict:
        """해외 주식의 상세 정보를 조회하여
        주요 지표(PER, PBR, EPS, BPS)만 저장 후 반환
        참조. https://apiportal.koreainvestment.com/apiservice/apiservice-oversea-stock-quotations#L_abc66a03-8103-4f6d-8ba8-450c2b935e14

        Args:
            market_code (str): 거래소 코드
                HKS : 홍콩
                NYS : 뉴욕
                NAS : 나스닥
                AMS : 아멕스
                TSE : 도쿄
                SHS : 상해
                SZS : 심천
                SHI : 상해지수
                SZI : 심천지수
                HSX : 호치민
                HNX : 하노이
                BAY : 뉴욕(주간)
                BAQ : 나스닥(주간)
                BAA : 아멕스(주간)
            stock_code (str): 종목 코드 (예: 'AAPL')
        """
        # 시장 코드 조회
        market_code = self.resolve_overseas_market_code(stock_code)

        # API 엔드포인트 설정
        path = "/uapi/overseas-price/v1/quotations/price-detail"
        url = f"{self.base_url}/{path}"

        # API 요청 헤더 설정 (인증 정보 포함)
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "HHDFS76200200",  # 해외주식 현재가상세 조회용 TR_ID
            "custtype": self.config["customer_type"],
        }

        params = {"AUTH": "", "EXCD": market_code, "SYMB": stock_code}

        time.sleep(0.1)  # API 호출 제한 방지를 위한 지연
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        # print("API 응답:", data)  # ✅ 요거 추가!

        if "output" not in data or not data["output"]:
            print("output 데이터 없음:", data)
            return pd.DataFrame()

        df = pd.DataFrame(
            data["output"], index=[datetime.now().strftime("%Y-%m-%d")]
        )  # 산출 시점 기록

        # extract columns perx, pbrx, epsx, bpsx from response
        df = df[["perx", "pbrx", "epsx", "bpsx"]]
        df.columns = ["PER", "PBR", "EPS", "BPS"]

        return df

    def get_ovs_financial_ratio(self, stock_code: str) -> pd.DataFrame:
        """
        해외 주식의 재무비율 정보를 조회하여 주요 지표(EPS, BPS, ROE)를 DataFrame으로 반환
        - ROE = EPS / BPS * 100 계산
        - stac_yymm은 조회 날짜 기준으로 YYYYMM 포맷 사용

        Args:
            stock_code (str): 종목 코드 (예: 'AAPL')

        Returns:
            pd.DataFrame: 컬럼은 stac_yymm, EPS, BPS, ROE
        """
        # 시장 코드 조회
        market_code = self.resolve_overseas_market_code(stock_code)

        # API 요청 기본 정보
        path = "/uapi/overseas-price/v1/quotations/price-detail"
        url = f"{self.base_url}/{path}"

        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "HHDFS76200200",
            "custtype": self.config["customer_type"],
        }

        params = {
            "AUTH": "",
            "EXCD": market_code,  # 거래소 코드 (예: 'NAS')
            "SYMB": stock_code,  # 종목 코드 (예: 'AAPL')
        }

        # API 요청
        time.sleep(1)
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        if "output" not in data or not data["output"]:
            print("해외 데이터 조회 실패:", data)
            return pd.DataFrame()

        # 데이터 추출 및 처리
        output = data["output"]
        eps = float(output.get("epsx", 0))
        bps = float(output.get("bpsx", 0))

        # ROE 계산 (BPS가 0일 경우 0 처리)
        roe = round((eps / bps) * 100, 2) if bps != 0 else 0.0

        # 기준일(stac_yymm) 추가 (현재 날짜 기준)
        stac_yymm = datetime.now().strftime("%Y%m")

        # DataFrame 생성 및 타입 변환
        df = pd.DataFrame(
            [{"stac_yymm": stac_yymm, "EPS": eps, "BPS": bps, "ROE(%)": roe}]
        )
        # 타입 안정성 확보 (명시적으로 숫자형 변환)
        numeric_columns = ["EPS", "BPS", "ROE(%)"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        return df

    def order_dom_stock(
        self,
        product_code: str,
        order_type: str,
        quantity: str,
        price: str,
        is_buy: bool = True,
    ) -> dict:
        """국내 주식을 주문하는 메서드

        Args:
            account_no (str): 계좌번호 앞 8자리 (예: '12345678')
            product_code (str): 종목 코드 6자리 (예: '005930' - 삼성전자)
            order_type (str): 주문 유형 (예: '00' - 지정가, '01' - 시장가)
            quantity (str): 주문 수량 (문자열 형식, 예: '10')
            price (str): 주문 단가 (문자열 형식, 지정가 주문 시 필수, 시장가 주문 시 '0' 입력)
            is_buy (bool): 매수(True) 또는 매도(False), 기본값은 매수

        Returns:
            dict: 주문 결과 (성공 시 주문번호 반환)

        Process:
            1. API 엔드포인트 URL과 인증 헤더 설정
            2. 주문 요청 파라미터 설정 후 API 요청 (POST 방식)
            3. 응답 데이터를 JSON 형태로 반환
        """
        # API 엔드포인트 설정
        path = "/uapi/domestic-stock/v1/trading/order-cash"
        url = f"{self.base_url}/{path}"

        # 주문 유형에 따른 거래 ID 선택
        tr_id = "TTTC0012U" if is_buy else "TTTC0011U"

        # API 요청 헤더 설정
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": self.config["customer_type"],
        }

        # 주문 요청 본문 설정
        payload = {
            "CANO": self.config["account"]["number"],  # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": self.config["account"][
                "product_code"
            ],  # 계좌상품코드 (뒤 2자리)
            "PDNO": product_code,  # 종목 코드 (6자리)
            "ORD_DVSN": order_type,  # 주문 구분 코드 (지정가, 시장가 등)
            "ORD_QTY": quantity,  # 주문 수량 (문자열)
            "ORD_UNPR": price,  # 주문 단가 (지정가 주문 시 필수, 시장가 주문 시 '0')
        }

        time.sleep(0.1)  # API 호출 제한 방지를 위한 지연
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        return data  # 주문 결과 반환

    def order_ovs_stock(
        self,
        market_code: str,
        product_code: str,
        quantity: str,
        price: str,
        is_buy: bool = True,
    ) -> dict:
        """해외 주식을 주문하는 메서드

        Args:
            market_code (str): 거래소 코드 (예: 'NAS' - 나스닥, 'NYS' - 뉴욕)
            product_code (str): 종목 코드 (예: 'TSLA' - 테슬라)
            quantity (str): 주문 수량 (문자열 형식, 예: '10')
            price (str): 주문 단가 (문자열 형식, 지정가 주문 시 필수, 시장가 주문 시 '0' 입력)
            is_buy (bool): 매수(True) 또는 매도(False), 기본값은 매수

        Returns:
            dict: 주문 결과 (성공 시 주문번호 반환)

        Process:
            1. API 엔드포인트 URL과 인증 헤더 설정
            2. 주문 요청 파라미터 설정 후 API 요청 (POST 방식)
            3. 응답 데이터를 JSON 형태로 반환
        """
        # API 엔드포인트 설정
        path = "/uapi/overseas-stock/v1/trading/order"
        url = f"{self.base_url}/{path}"

        # 매수/매도에 따른 거래 ID 선택 (모의투자 미국 기준)
        tr_id = "VTTT1002U" if is_buy else "VTTT1001U"

        # API 요청 헤더 설정
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": self.config["customer_type"],
        }

        # 주문 요청 본문 설정
        payload = {
            "CANO": self.config["account"]["number"],  # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": self.config["account"][
                "product_code"
            ],  # 계좌상품코드 (뒤 2자리)
            "OVRS_EXCG_CD": market_code,  # 거래소 코드 (NYS: 뉴욕, NAS: 나스닥 등)
            "PDNO": product_code,  # 종목 코드 (6자리)
            "ORD_QTY": quantity,  # 주문 수량
            "OVRS_ORD_UNPR": price,  # 1주당 가격, 시장가의 경우 1주당 가격을 공란으로 비우지 않음 "0"으로 입력
            "SLL_TYPE": "" if is_buy else "00",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",
        }

        time.sleep(0.1)  # API 호출 제한 방지를 위한 지연
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        return data  # 주문 결과 반환

if __name__ == "__main__":
    # 차트 데이터 저장 테스트
    kis = Kis()
    
    query_start = datetime(2024, 1, 1) # 조회 시작 시점
    fetch_end = datetime.now() # 조회 종료 시점
    period_code = 0 # 조회 주기: 일봉

    ovs_chart = kis.get_overseas_period_chart(
        "NVDA", 
        fetch_end.strftime("%Y%M%d"), 
        period_code = period_code
    )

    # csv로 차트 데이터 저장
    with open(Path(__file__).resolve().parent / 'NVDA_chart.csv', 'w') as f:
        ovs_chart.to_csv(f)

    print(f"데이터 저장 완료: {len(ovs_chart)}건")
