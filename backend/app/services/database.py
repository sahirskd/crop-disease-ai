from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings
from datetime import datetime

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionModel(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=True)
    plant = Column(String)
    disease = Column(String)
    confidence = Column(Float)
    is_healthy = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(db: Session, data: dict):
    record = PredictionModel(**data)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_history(db: Session, limit: int = 20):
    return (
        db.query(PredictionModel)
        .order_by(PredictionModel.created_at.desc())
        .limit(limit)
        .all()
    )
