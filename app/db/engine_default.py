from contextlib import contextmanager
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from app.helpers.error_helper import log_error
from app.core.config import settings  # Import settings

engine = create_engine(settings.DATABASE_URL)  # Use DATABASE_URL from settings
metadata = MetaData()
metadata.bind = engine

Base = declarative_base(metadata=metadata)

Session = sessionmaker(bind=engine)


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
    except Exception as e:
        log_error(e)
        session.rollback()
        raise
    finally:
        session.close()
