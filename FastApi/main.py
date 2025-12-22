from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import joblib
import re
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
import random
from faker import Faker
import os
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, CheckConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import func



app = FastAPI(title="prediction model")




class Tweet(BaseModel):
    airline_sentiment_confidence: float
    airline: str
    negativereason: Optional[str]
    tweet_created: str 
    text: str

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float


# DATABASE_URL = os.getenv(
#     "DATABASE_URL",
#     "postgresql+psycopg2://aerostream:aerostream@postgres:5432/aerostream",
# )
DATABASE_URL="postgresql+psycopg2://aerostream:aerostream@postgres:5432/aerostream"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base = declarative_base()


class TweetDB(Base):
    __tablename__ = "tweets"

    id = Column(Integer, primary_key=True, index=True)
    airline_sentiment_confidence = Column(Float, nullable=False)
    airline = Column(String(50), nullable=False)
    negativereason = Column(String(100), nullable=True)
    tweet_created = Column(DateTime(timezone=True), nullable=False)
    text = Column(Text, nullable=False)
    prediction = Column(String(10), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "airline_sentiment_confidence >= 0 AND airline_sentiment_confidence <= 1",
            name="confidence_range",
        ),
    )












fake = Faker()
Faker.seed(42)

AIRLINES = ['Virgin America', 'United', 'Southwest', 'Delta', 'US Airways', 'American']
SENTIMENTS = ['neutral', 'positive', 'negative']
NEGATIVE_REASONS = [
    None,
    'Bad Flight',
    "Can't Tell",
    'Late Flight',
    'Customer Service Issue',
    'Flight Booking Problems',
    'Lost Luggage',
    'Flight Attendant Complaints',
    'Cancelled Flight',
    'Damaged Luggage',
    'longlines'
]








# BASE_DIR = Path(__file__).resolve().parent.parent
# MODEL_PATH = BASE_DIR / "models" / "airline_sentiment_svm.pkl"
# ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

model = joblib.load("./models/airline_sentiment_svm.pkl")
encoder = joblib.load("./models/label_encoder.pkl")
embedding_model = SentenceTransformer("all-MiniLM-L12-v2")













def preprocess_text(text):
    """Clean and preprocess the input text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()
    return text









def predict_setiment(text, model, encoder, embedding_model):
    
    try:
        cleaned_text = preprocess_text(text)
        embedding = embedding_model.encode([cleaned_text], normalize_embeddings=True)
        
        prediction = model.predict(embedding)[0]
        probas = model.predict_proba(embedding)[0]  # shape: (n_classes,)
        class_idx = list(model.classes_).index(prediction)
        confidence = float(probas[class_idx])

        sentiment = encoder.inverse_transform([prediction])[0]
        return {"label": sentiment, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    
    
    
   
   
   
   
   
   
   
   
   
    
def generate_tweet() -> Tweet:
    airline = random.choice(AIRLINES)
    sentiment = random.choices(
        SENTIMENTS,
        weights=[0.3, 0.25, 0.45],
        k=1
    )[0]
    
    confidence = round(random.uniform(0.5, 1.0), 3) 
    
    
    
    if sentiment == 'neutral':
        confidence = round(random.uniform(0.3, 0.7), 3)

    negativereason = None
    if sentiment == 'negative':
        negativereason = random.choice(NEGATIVE_REASONS[1:]) 
    elif sentiment == 'neutral':
        negativereason = random.choice(NEGATIVE_REASONS) 



   
    handles = {
        'Virgin America': '@VirginAmerica',
        'United': '@united',
        'Southwest': '@SouthwestAir',
        'Delta': '@Delta',
        'US Airways': '@USAirways',
        'American': '@AmericanAir'
    }
    handle = handles[airline]
    
    
    

    if sentiment == 'positive':
        texts = [
            f"{handle} Great service today — flight was on time and crew was amazing! ✈️👏",
            f"Shoutout to {handle} for upgrading me last minute. You made my day!",
            f"Smooth flight with {handle} — love the new seats and in-flight snacks. 🍪",
        ]
        
        
        
        
    elif sentiment == 'negative':
        texts = [
            f"{handle} why are your first fares in May over three times more than other carriers when all seats are available to select???",
            f"{handle} flight delayed 4 hours with no updates. Terrible communication. #disappointed",
            f"{handle} lost my luggage AGAIN. This is the third time this year. Unacceptable.",
            f"{handle} customer service hung up on me. What kind of support is that?!",
            f"{handle} 2-hour line at check-in for pre-paid bags. Ridiculous inefficiency.",
        ]
        
        
        
        
    else: 
        texts = [
            f"Flying with {handle} this afternoon from JFK to LAX.",
            f"Average experience with {handle}. On time, but seat was a bit tight.",
            f"My {handle} flight is scheduled for 8:45 AM tomorrow.",
            f"Currently boarding flight 452 with {handle}.",
        ]

    text = random.choice(texts)
    
    
    
    
    if random.random() < 0.3:
        text += " " + fake.sentence(nb_words=6).rstrip(".")

    tweet_created = datetime.now(timezone.utc).isoformat()

    return Tweet(
        airline_sentiment_confidence=confidence,
        airline=airline,
        negativereason=negativereason,
        tweet_created=tweet_created,
        text=text
    )
       
  






def conn_db():
    Base.metadata.create_all(bind=engine)
    return SessionLocal
    

def load_tweets(SessionLocal, batch_size: int = 10) -> int:
    batch_size = min(max(int(batch_size), 1), 100)
    db = SessionLocal()
    inserted = 0
    try:
        for _ in range(batch_size):
            t = generate_tweet()
            pred = predict_setiment(t.text, model, encoder, embedding_model)
            db.add(
                TweetDB(
                    airline_sentiment_confidence=t.airline_sentiment_confidence,
                    airline=t.airline,
                    negativereason=t.negativereason,
                    tweet_created=datetime.fromisoformat(t.tweet_created),
                    text=t.text,
                    prediction=pred["label"],
                )
            )
            inserted += 1
        db.commit()
        return inserted
    finally:
        db.close()

 

  
  
  
  
    


@app.get("/")
def test():
    return "hello world!"



    
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    
    text = request.text
    
    return predict_setiment(text, model, encoder, embedding_model)



    
    
@app.get("/batch", response_model=List[Tweet])
def get_microbatch(batch_size: int = 10):
    
    if not (1 <= batch_size <= 100):
        batch_size = min(max(batch_size, 1), 100)  
    return [generate_tweet() for _ in range(batch_size)]




@app.get("/db/conn/create")
def check_conn(batch_size: int = 10):
    SessionLocal = conn_db()
    inserted = load_tweets(SessionLocal, batch_size=batch_size)
    return {"inserted": inserted}




