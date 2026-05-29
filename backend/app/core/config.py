from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    MODEL_PATH: str = "experiments/best_model.pt"
    METADATA_PATH: str = "experiments/metadata.json"
    CLASS_NAMES_PATH: str = "experiments/class_names.json"
    MODEL_NAME: str = "efficientnet_b3"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "https://your-frontend.vercel.app"]
    DATABASE_URL: str = "sqlite:///./cropsense.db"  # swap for Supabase postgres in prod

    class Config:
        env_file = ".env"


settings = Settings()
