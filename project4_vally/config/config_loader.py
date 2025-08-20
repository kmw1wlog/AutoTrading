from openai import OpenAI
from pathlib import Path
import yaml

# 경로 세팅 정확한지 확인
project_root = Path(__file__).resolve().parent.parent

config_path = project_root / "config" / "config.yaml"
#  config.yaml 파일 로드
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OPENAI_API_KEY = config.get("OPENAI_API_KEY")
DART_API_KEY = config.get("DART_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
