# Filename: main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pytesseract
from PIL import Image
from transformers import pipeline
from sqlalchemy import create_engine, Column, String, Integer, Float
from sqlalchemy.orm import sessionmaker, declarative_base
import pickle
import pandas as pd
import uvicorn

# -------------------------------
# Configuration and Setup
# -------------------------------

# Database Configuration
DATABASE_URL = "sqlite:///./postal_db.db"  # SQLite for simplicity. Change to PostgreSQL if needed.

# Initialize Database
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define Database Models
class PostalCode(Base):
    _tablename_ = 'postal_codes'
    id = Column(Integer, primary_key=True, index=True)
    pincode = Column(String, unique=True, index=True, nullable=False)
    post_office = Column(String, nullable=False)
    delivery = Column(String, nullable=False)
    district = Column(String, nullable=False)
    state = Column(String, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(title="AI-Powered Postal Delivery System")

# Load NLP Model
address_parser = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# -------------------------------
# Pydantic Models for API
# -------------------------------

class AddressInput(BaseModel):
    address: str

class ValidationResponse(BaseModel):
    valid: bool
    post_office: str
    delivery: str

class UpdatePostalCodeInput(BaseModel):
    pincode: str
    post_office: str
    delivery: str
    district: str
    state: str
    latitude: float = None
    longitude: float = None

# -------------------------------
# Utility Functions
# -------------------------------

def populate_database_from_csv(csv_path: str):
    """Populate the database with postal data from the given CSV."""
    db = SessionLocal()
    try:
        data = pd.read_csv(csv_path)
        for _, row in data.iterrows():
            existing_entry = db.query(PostalCode).filter(PostalCode.pincode == str(row['Pincode'])).first()
            if not existing_entry:
                entry = PostalCode(
                    pincode=str(row['Pincode']),
                    post_office=row['OfficeNam'],
                    delivery=row['Delivery'],
                    district=row['District'],
                    state=row['StateNam'],
                    latitude=row['Latitude'],
                    longitude=row['Longitude']
                )
                db.add(entry)
        db.commit()
    finally:
        db.close()
    print("Database populated from CSV.")

def validate_pincode(pincode: str, db_session):
    """Validate PIN code against the database."""
    postal_entry = db_session.query(PostalCode).filter(PostalCode.pincode == pincode).first()
    if postal_entry:
        return True, postal_entry
    else:
        return False, None

# -------------------------------
# API Endpoints
# -------------------------------

@app.on_event("startup")
async def startup_event():
    """Populate the database with initial data from the CSV."""
    populate_database_from_csv("coimbature_df.csv")  # Ensure the dataset file is available.

@app.post("/validate_pincode", response_model=ValidationResponse)
async def validate_pincode_endpoint(pincode: str):
    """Endpoint to validate a PIN code."""
    try:
        db = SessionLocal()
        is_valid, postal_entry = validate_pincode(pincode, db)
        db.close()
        if not is_valid:
            raise HTTPException(status_code=404, detail="PIN code not found.")
        return {
            "valid": True,
            "post_office": postal_entry.post_office,
            "delivery": postal_entry.delivery
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_postal_code")
async def add_postal_code(input: UpdatePostalCodeInput):
    """Endpoint to add or update a postal code in the database."""
    try:
        db = SessionLocal()
        existing_entry = db.query(PostalCode).filter(PostalCode.pincode == input.pincode).first()
        if existing_entry:
            existing_entry.post_office = input.post_office
            existing_entry.delivery = input.delivery
            existing_entry.district = input.district
            existing_entry.state = input.state
            existing_entry.latitude = input.latitude
            existing_entry.longitude = input.longitude
        else:
            new_entry = PostalCode(
                pincode=input.pincode,
                post_office=input.post_office,
                delivery=input.delivery,
                district=input.district,
                state=input.state,
                latitude=input.latitude,
                longitude=input.longitude
            )
            db.add(new_entry)
        db.commit()
        db.close()
        return {"message": "Postal code added/updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/postal_code/{pincode}")
async def get_postal_code_info(pincode: str):
    """Get details of a postal code."""
    try:
        db = SessionLocal()
        postal_entry = db.query(PostalCode).filter(PostalCode.pincode == pincode).first()
        db.close()
        if not postal_entry:
            raise HTTPException(status_code=404, detail="Postal code not found.")
        return {
            "pincode": postal_entry.pincode,
            "post_office": postal_entry.post_office,
            "delivery": postal_entry.delivery,
            "district": postal_entry.district,
            "state": postal_entry.state,
            "latitude": postal_entry.latitude,
            "longitude": postal_entry.longitude
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Run the Application
# -------------------------------

if _name_ == "_main_":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)