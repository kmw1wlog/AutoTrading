from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import json
import sys
from typing import Literal

# project 루트 디렉토리를 파이썬 path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from kis.kis import Kis

class ChartFetcher:
    """
    차트 데이터를 관리하는 클래스
    - 국내(dom) 및 해외(ovs) 시장 데이터 지원
    - 데이터 저장, 병합, 필터링, API 호출 관리
    """
    def __init__(self):
        """
        설정 초기화 및 데이터 디렉토리 구성
        """
        # settings.json 로드
        settings_path = Path(__file__).parent / "settings.json"
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        # 차트 타입별 수집 시작 기준일 생성
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.collection_start_times = {
            chart_type: today
            - timedelta(days=settings["chart_collection"][chart_type]["days_ago"])
            for chart_type in ["min", "daily", "weekly"]
        }

        # KIS API 인스턴스 생성
        self.kis = Kis()

        # 데이터 저장 경로 초기화
        self.data_path = Path(__file__).resolve().parent.parent / "data"
        self.data_charts_path = self.data_path / "charts"
        self._init_data_directories()

    def _init_data_directories(self):
        """
        데이터 저장 디렉토리 생성
        (국내(dom) / 해외(ovs) 구분 및 주기(min/daily/weekly) 디렉토리)
        """
        for market in ["dom", "ovs"]:
            market_path = self.data_charts_path / market
            for interval in ["min", "daily", "weekly"]:
                interval_path = market_path / interval
                interval_path.mkdir(parents=True, exist_ok=True)

    def get_chart(
        self,
        market: Literal["dom", "ovs"],
        stock_code: str,
        start_datetime: datetime = datetime.now() - timedelta(days=365 * 3),
        end_datetime: datetime = datetime.now(),
        interval: Literal["min", "daily", "weekly"] = "daily",
        overwrite: bool = False,
    ) -> pd.DataFrame:

        now = datetime.now()
        today = now

        # 오늘을 제외한 일/주봉 데이터 수집
        if interval in ["daily", "weekly"] and end_datetime >= today:
            end_datetime = datetime.combine(
                today - timedelta(days=1), datetime.min.time()
            )

        # 1. 기존 데이터 로드
        existing_data = self._read_chart_data(market, stock_code, interval)

        if overwrite:
            print("[INFO] Overwrite 모드 → 전체 재수집")
            query_start = self.collection_start_times[interval]
            data = pd.DataFrame()  # 빈 데이터에서 시작
        else:
            if existing_data is not None and not existing_data.empty:
                existing_data.index = pd.to_datetime(existing_data.index)
                last_date = existing_data.index[-1]

                if last_date >= end_datetime:
                    print("[INFO] 기존 데이터 최신 → 기존 데이터 반환")
                    return existing_data.loc[start_datetime:end_datetime]

                query_start = last_date + self._get_interval_timedelta(interval)
                data = existing_data.copy()
            else:
                print("[INFO] 기존 차트 데이터 없음 → 최초 수집")
                query_start = self.collection_start_times[interval]
                data = pd.DataFrame()

        # 2. API 호출 → 데이터 수집
        fetch_end = end_datetime
        all_chunks = []

        while query_start <= fetch_end:
            # 국내
            if market == "dom":
                if interval == "min":
                    chart_chunk = self.kis.get_minute_chart_past(
                        stock_code,
                        date=fetch_end.strftime("%Y%m%d"),
                        time_from=fetch_end.strftime("%H%M%S"),
                    )
                elif interval in ["daily", "weekly"]:
                    period_code = "D" if interval == "daily" else "W"
                    chart_chunk = self.kis.get_period_chart(
                        stock_code,
                        query_start.strftime("%Y%m%d"),
                        fetch_end.strftime("%Y%m%d"),
                        period_code=period_code,
                    )
                else:
                    raise ValueError(f"Invalid interval: {interval}")

            # 해외
            else:
                if interval == "min":
                    chart_chunk = self.kis.get_overseas_minute_chart(
                        stock_code, interval=1, include_prev_day=True
                    )
                elif interval == "daily":
                    chart_chunk = self.kis.get_overseas_period_chart(
                        stock_code,
                        fetch_end.strftime("%Y%m%d"),
                        period_code="0",
                    )
                else:
                    raise ValueError(f"Invalid interval: {interval}")

            # 수집 실패 (빈 데이터)
            if chart_chunk is None or chart_chunk.empty:
                print(
                    f"[WARN] {fetch_end.date()} ~ {query_start.date()} 차트 데이터 없음"
                )
                break  # 재시도하거나 continue 가능하지만 간단히 break로 처리

            # 데이터 정렬 후 저장 준비
            chart_chunk.sort_index(inplace=True, ascending=False)  # 시간 순으로 정렬
            all_chunks.append(chart_chunk)

            # 다음 수집 시작점
            last_ts = chart_chunk.index[-1]
            fetch_end = last_ts - self._get_interval_timedelta(interval)

        # 3. 신규 데이터 병합
        if all_chunks:
            fetched_data = pd.concat(all_chunks).sort_index()

            if not data.empty:
                new_data = fetched_data[fetched_data.index > data.index.max()]
                combined_data = pd.concat([data, new_data]).sort_index()
            else:
                combined_data = fetched_data

            self._save_chart_data(combined_data, market, stock_code, interval)

            return combined_data.loc[start_datetime:end_datetime]

        # 4. 수집된 게 없어도 기존 데이터 반환
        if not data.empty:
            print("[WARN] 신규 데이터 없음 → 기존 데이터 반환")
            return data.loc[start_datetime:end_datetime]

        # 5. 신규, 기존 데이터 모두 없음
        print("[ERROR] 최종 반환할 데이터 없음!")
        return pd.DataFrame()

    def _get_interval_timedelta(self, interval: str) -> timedelta:
        if interval == "min":
            return timedelta(minutes=1)
        elif interval == "daily":
            return timedelta(days=1)
        elif interval == "weekly":
            return timedelta(weeks=1)
        else:
            raise ValueError(f"Invalid interval: {interval}")

    def _read_chart_data(
        self, market: str, stock_code: str, interval: str
    ) -> pd.DataFrame:
        """
        기존 차트 데이터 파일을 읽어오기
        """
        save_dir = self.data_charts_path / market / interval
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / f"{stock_code}.csv"
        if not file_path.exists():
            return None

        df = pd.read_csv(file_path, index_col="datetime", parse_dates=True)

        return df

    def _save_chart_data(
        self,
        new_df: pd.DataFrame,
        market: str,
        stock_code: str,
        interval: str,
        overwrite: bool = False,
    ):
        """
        병합된 차트 데이터를 파일에 저장한다.
        """
        save_dir = self.data_charts_path / market / interval
        save_dir.mkdir(parents=True, exist_ok=True)

        file_path = save_dir / f"{stock_code}.csv"

        if overwrite:
            merged_df = new_df
        else:
            if file_path.exists():
                existing_df = pd.read_csv(
                    file_path, index_col="datetime", parse_dates=True
                )
                merged_df = pd.concat([existing_df, new_df])
            else:
                merged_df = new_df

        # 중복 인덱스 제거 및 정렬
        merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
        merged_df.sort_index(inplace=True)

        # 미완성 캔들 필터링
        filtered_df = self._filter_incomplete_candles(merged_df, interval, market)

        # 저장
        filtered_df.to_csv(file_path)

        return filtered_df

    def _filter_incomplete_candles(
        self, df: pd.DataFrame, interval: str, market: str
    ) -> pd.DataFrame:
        """
        각 interval별 안전하게 완성된 캔들스틱 시점을 반환
            오늘 데이터는 장 마감 후에만 저장
        예) 분봉: 현재 시각의 분 정보가 완성된 이전 분까지,
            일봉: 오늘 날짜는 아직 미완성이므로 오늘 0시를 safe cutoff로 함,
        """
        now = datetime.now()
        today_date = now.date()

        if interval == "min":
            safe_cutoff = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
            return df[df.index <= safe_cutoff]

        elif interval == "daily":
            # 오늘 데이터가 있다면 검사
            if today_date in df.index.date:
                if market == "dom":
                    market_close = now.replace(
                        hour=15, minute=30, second=0, microsecond=0
                    )
                elif market == "ovs":
                    market_close = now.replace(
                        hour=6, minute=0, second=0, microsecond=0
                    )
                else:
                    raise ValueError(f"Unknown market type: {market}")

                if now < market_close:
                    # 마감 전이면 오늘 데이터 제거
                    # print(f"[INFO] {market.upper()} 시장 마감 전 → 오늘 데이터 제외")
                    return df[df.index.date < today_date]
                else:
                    # 마감 후면 오늘 데이터 포함 저장
                    # print(f"[INFO] {market.upper()} 시장 마감 후 → 오늘 데이터 포함")
                    return df

            else:
                return df

        elif interval == "weekly":
            # 주봉은 주말 지나야 확정! (일단 월요일 00시 기준으로)
            safe_week_cutoff = datetime.combine(
                (now - timedelta(days=now.weekday())).date(), datetime.min.time()
            )
            return df[df.index < safe_week_cutoff]

        else:
            return df


if __name__ == "__main__":
    chart_fetcher = ChartFetcher()
    df = chart_fetcher.get_chart(
        "ovs",
        "MRNA",
        datetime.now() - timedelta(days=365 * 3),
        datetime.now(),
        "daily",
        overwrite=False,
    )
    print(df.head())