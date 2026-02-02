import os
from pathlib import Path

JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

# Database path - use absolute path to avoid issues with working directory
_db_path = os.getenv("DB_URL")
if _db_path:
    DB_URL = _db_path
else:
    # Default: create app.db in the project root
    project_root = Path(__file__).parent.parent
    db_file = project_root / "app.db"
    DB_URL = f"sqlite:///{db_file.absolute()}"

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")





