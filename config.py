import os

class Config:
    # Kafka Configuration
    KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'localhost:9092')
    
    # Model Configurations
    LANGUAGE_MODEL = 'gpt2-medium'
    
    # Data Collection Settings
    MAX_USER_INTERACTIONS = 100
    
    # Recommendation Settings
    RECOMMENDATION_BATCH_SIZE = 5
    
    # Emotion Detection Thresholds
    EMOTION_CONFIDENCE_THRESHOLD = 0.5

# Environment-specific configurations
class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False