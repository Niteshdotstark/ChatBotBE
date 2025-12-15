from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os
from pathlib import Path

# Define Base FIRST
Base = declarative_base()


# Load environment variables
env_path = Path(".").resolve() / "env"  # Point to your env file
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URI")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()