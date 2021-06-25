from pathlib import Path


class Config:

    BASE_DIR = Path(__file__).resolve().parent.parent

    # Data handlers.
    DATA_DIR = BASE_DIR / "__DATA__"
    CAR_POSE_DIR = DATA_DIR / "car_pos"
    CAR_POSE_DATA_DIR = CAR_POSE_DIR / "data"
    CAR_POSE_JSON_DIR = CAR_POSE_DIR / "json"

    # Model handlers.
    MODEL_DIR = BASE_DIR / "__MODEL__"

    # Logs handlers.
    LOGS_DIR = BASE_DIR / "__LOGS__"


config = Config()