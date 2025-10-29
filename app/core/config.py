import os
from dotenv import load_dotenv
from pydantic import BaseSettings

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
load_dotenv(os.path.join(BASE_DIR, '.env'))


class Settings(BaseSettings):
    PROJECT_NAME = os.getenv('PROJECT_NAME', 'FASTAPI BASE')
    SECRET_KEY = os.getenv('SECRET_KEY', '')
    API_PREFIX = ''
    BACKEND_CORS_ORIGINS = ['http://localhost:5173', 'http://35.197.136.79:5173']
    DATABASE_URL = os.getenv('SQL_DATABASE_URL', '')
    ACCESS_TOKEN_EXPIRE_SECONDS: int = 60 * 60 * 24 * 7  # Token expired after 7 days
    SECURITY_ALGORITHM = 'HS256'
    LOGGING_CONFIG_FILE = os.path.join(BASE_DIR, 'logging.ini')
    ACCESS_TOKEN_PUBLIC_KEY = os.getenv('ACCESS_TOKEN_PUBLIC_KEY')
    MODEL = os.getenv('MODEL')
    ELASTICSEARCH_ENPOINT = os.getenv('ELASTICSEARCH_ENPOINT')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
    OPENAI_API_KEY_EMBEDDING = os.getenv('OPENAI_API_KEY_EMBEDDING')
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    MODEL_PATH = os.getenv('MODEL_PATH')
    UPLOAD_FILE_DIRECTORY=os.getenv('UPLOAD_FILE_DIRECTORY')


settings = Settings()
