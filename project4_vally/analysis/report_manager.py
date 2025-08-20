from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

class ReportManager:
    def __init__(
        self,
        report_root: Path,
        report_type: Literal["fundamental", "news", "technical"],
        settings: dict,
    ):
        """
        Args:
            report_root (Path): report 저장 루트 경로 (Path 객체)
            report_type (Literal["fundamental", "news", "technical"]): 'fundamental' / 'news' / 'technical'
            settings (dict): analysis_settings.json 딕셔너리 전체 or 필요한 부분

        Returns:
            None
        """
        self.report_root = report_root / report_type
        self.report_root.mkdir(parents=True, exist_ok=True)

        # settings는 이미 update_interval만 들어오니까 바로 참조
        self.update_interval = settings["update_interval"]

        self.update_delta = timedelta(
            days=self.update_interval.get("days", 0),
            hours=self.update_interval.get("hours", 0),
        )

    def get_latest_report(
        self, market: str, stock_code: str, extension="md"
    ) -> tuple[datetime, Path] | None:
        """
        해당 종목의 최신 리포트 반환
        Args:

        Returns:
            tuple[datetime, Path] | None: 최신 리포트 시간과 경로, 없으면 None
        """
        report_dir = self.report_root / market
        if not report_dir.exists():
            return None

        files = list(report_dir.glob(f"{stock_code}_*.{extension}"))
        if not files:
            return None

        # 파일명에서 날짜 추출 후 최신 정렬
        def extract_datetime(file):
            filename = file.stem  # 확장자 제외
            try:
                date_str = filename.split("_")[1] + filename.split("_")[2]
                return datetime.strptime(date_str, "%Y%m%d%H%M%S")
            except (IndexError, ValueError):
                return datetime.min

        files.sort(key=extract_datetime, reverse=True)
        latest_file = files[0]
        latest_time = extract_datetime(latest_file)

        return latest_time, latest_file

    def is_update_required(self, latest_time: datetime) -> bool:
        """
        최신 보고서가 업데이트 기준을 넘었는지 여부
        """
        return (datetime.now() - latest_time) > self.update_delta

    def save_report(
        self, market: str, stock_code: str, content: str, extension="md"
    ) -> Path:
        """
        리포트 저장

        Returns:
            Path: 저장된 파일 경로(Path)
        """
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{stock_code}_{now_str}.{extension}"

        report_dir = self.report_root / market
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / report_filename
        report_path.write_text(content, encoding="utf-8")

        return report_path
