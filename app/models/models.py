from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, BigInteger
from sqlalchemy.sql import func

from app.db.engine_default import Base


class Category(Base):
    __tablename__ = 'categories'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    parent_category_id = Column(Integer, ForeignKey('categories.id'))
    organization_id = Column(Integer, ForeignKey('organizations.id'))
    department_id = Column(Integer, ForeignKey('departments.id'))
    created_by = Column(String(30), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Log(Base):
    __tablename__ = 'logs'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    action = Column(String(30))
    request_data = Column(String(255))
    created_at = Column(DateTime)


class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    category_id = Column(Integer, ForeignKey('categories.id'))
    status = Column(Integer, nullable=False)
    created_by = Column(String(30), nullable=False)
    type = Column(String(30), nullable=False)
    total_page = Column(Integer)
    description = Column(String(255))
    file_path = Column(String(255))
    file_id = Column(String(255))
    storage_capacity = Column(Integer)
    storage_unit = Column(String(30))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    access_type = Column(Integer)
    organization_id = Column(Integer)
    dept_id = Column(Integer)
    photo_id = Column(Text, nullable=True)
    password = Column(Text)


class Comment(Base):
    __tablename__ = 'comments'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String(255))
    content = Column(Text)
    parent_comment_id = Column(Integer, ForeignKey('comments.id'))
    status = Column(Integer, default=1, nullable=False)
    doc_id = Column(Integer, ForeignKey('documents.id'))
    # multimedia_id = Column(Integer, ForeignKey('multimedia.id'))


class Notification(Base):
    __tablename__ = 'notifications'

    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    content = Column(Text)


class Organization(Base):
    __tablename__ = 'organizations'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(String(255))
    status = Column(Integer, default=1, nullable=False)
    limit_data = Column(BigInteger)
    data_used = Column(BigInteger)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(50))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    updated_by = Column(String(50))
    is_openai = Column(Boolean, default=False)
    limit_token = Column(BigInteger)
    token_used = Column(BigInteger)

class SearchHistory(Base):
    __tablename__ = 'search_history'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    title = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserDocument(Base):
    __tablename__ = 'user_documents'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)  # new field
    type = Column(Integer)
    status = Column(Integer, default=1, nullable=False)
    decentralized_by = Column(String(30))
    updated_at = Column(DateTime)
    viewed_at = Column(DateTime)
    move_to_trash_at = Column(DateTime)
    created_at = Column(DateTime)


class UserNotification(Base):
    __tablename__ = 'user_notifications'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    noti_id = Column(Integer, ForeignKey('notifications.id'), nullable=False)
    title = Column(String(255))
    content = Column(String(500))
    status = Column(Integer)


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    first_name = Column(String(30), nullable=False)
    last_name = Column(String(30), nullable=False)
    email = Column(String(30), nullable=False, unique=True)
    password = Column(Text)
    gender = Column(Boolean)
    status = Column(Integer, default=1, comment="1 - active, 2 - deactive")
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    is_organization_manager = Column(Boolean)
    organization_id = Column(Integer, ForeignKey('organizations.id'))
    is_social = Column(Boolean, default=False)
    dept_id = Column(Integer)
    is_dept_manager = Column(Boolean)


class Department(Base):
    __tablename__ = 'departments'

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    created_by = Column(String(100))
    created_at = Column(DateTime)
    status = Column(Integer)


class PrivateDoc(Base):
    __tablename__ = 'private_docs'

    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    status = Column(Integer, default=1, comment="1 - shared, 2 - removed")


class StarredDoc(Base):
    __tablename__ = 'starred_docs'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    doc_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    status = Column(Integer, comment='1-star, 2-unstar')


class FileUpload(Base):
    __tablename__ = 'file_uploads'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    file_path = Column(String(255), nullable=False)