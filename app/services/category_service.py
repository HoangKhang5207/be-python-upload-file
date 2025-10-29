# app/services/category_service.py
from app.db.engine_default import session_scope
from app.models.models import Category


def get_all_categories():
    with session_scope() as session:
        categories = session.query(Category).all()
        return [category.__dict__ for category in categories]