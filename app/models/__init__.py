# Import all the models, so that Base has them before being
# imported by Alembic
# from app.models.model_base import Base  # noqa
# from app.models.model_user import User  # noqa

from app.models.model_base import Base

# Import TẤT CẢ các model classes từ models.py
from app.models.models import *

# Đảm bảo Base được export
__all__ = ['Base']