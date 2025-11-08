from datetime import timedelta
import datetime
import uuid
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, BigInteger
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

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
    
    # Bổ sung relationship (tùy chọn nhưng nên có)
    documents = relationship("Document", back_populates="category")

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
    content = Column(Text, nullable=True) # Dùng cho UC-87 (OCR)
    category_id = Column(Integer, ForeignKey('categories.id'))
    status = Column(Integer, nullable=False, default=1) # 1=DRAFT (theo UC-39), 2=PROCESSING, 3=DONE, 0=DELETED
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
    access_type = Column(Integer) # 1=Public, 2=Org, 3=Dept, 4=Private
    organization_id = Column(Integer, ForeignKey('organizations.id')) # Bổ sung ForeignKey
    dept_id = Column(Integer, ForeignKey('departments.id')) # Bổ sung ForeignKey
    photo_id = Column(Text, nullable=True)
    password = Column(Text)
    
    # --- CÁC TRƯỜNG MỚI BỔ SUNG THEO UC-39 ---
    version = Column(String(20), default="1.0", nullable=False) # UC-39: version=1.0
    confidentiality = Column(String(50), default="INTERNAL", nullable=False) # UC-39: (PUBLIC/INTERNAL/LOCKED)
    tags_json = Column(JSONB, nullable=True) # UC-73: Lưu tags/keywords
    is_paid = Column(Boolean, default=False, nullable=False) # UC-85: Hỗ trợ access_type=paid
    
    # Bổ sung relationships (tùy chọn nhưng nên có)
    category = relationship("Category", back_populates="documents")
    organization = relationship("Organization", back_populates="documents")
    department = relationship("Department", back_populates="documents")
    user_documents = relationship("UserDocument", back_populates="document", cascade="all, delete-orphan")
    public_links = relationship("PublicLink", back_populates="document", cascade="all, delete-orphan")
    workflow_instances = relationship("WorkflowInstance", back_populates="document", cascade="all, delete-orphan")


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
    
    # Bổ sung relationships
    documents = relationship("Document", back_populates="organization")
    departments = relationship("Department", back_populates="organization")

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
    document_id = Column(Integer, ForeignKey('documents.id', ondelete="CASCADE"), nullable=False) # Thêm ondelete
    type = Column(Integer)
    status = Column(Integer, default=1, nullable=False)
    decentralized_by = Column(String(30))
    updated_at = Column(DateTime)
    viewed_at = Column(DateTime)
    move_to_trash_at = Column(DateTime)
    created_at = Column(DateTime)
    
    # Bổ sung relationships
    document = relationship("Document", back_populates="user_documents")
    user = relationship("User", back_populates="user_documents")


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
    dept_id = Column(Integer, ForeignKey('departments.id')) # Bổ sung ForeignKey
    is_dept_manager = Column(Boolean)
    
    # Bổ sung relationships
    user_documents = relationship("UserDocument", back_populates="user")
    workflow_instances = relationship("WorkflowInstance", back_populates="triggered_by")


class Department(Base):
    __tablename__ = 'departments'

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey('organizations.id'), nullable=False) # Bổ sung ForeignKey
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    created_by = Column(String(100))
    created_at = Column(DateTime)
    status = Column(Integer)
    
    # Bổ sung relationships
    organization = relationship("Organization", back_populates="departments")
    documents = relationship("Document", back_populates="department")


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
    
# --- BẢNG MỚI BỔ SUNG (UC-86, UC-84) ---

class PublicLink(Base):
    """
    Model lưu trữ liên kết công khai (UC-86)
    """
    __tablename__ = "public_links"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(255), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    expires_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now() + timedelta(hours=72)) # UC-86: 72 giờ
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    document = relationship("Document", back_populates="public_links")

class WorkflowInstance(Base):
    """
    Model ghi lại việc kích hoạt workflow (UC-84)
    """
    __tablename__ = "workflow_instances"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    workflow_name = Column(String(255), nullable=False)
    process_key = Column(String(255), nullable=True) # Key của BPMN
    status = Column(String(50), default="PENDING") # PENDING, APPROVED, REJECTED
    triggered_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    candidate_group = Column(String(255), nullable=True) # Nhóm/Phòng ban xử lý
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    document = relationship("Document", back_populates="workflow_instances")
    triggered_by = relationship("User") # Cần thêm relationship "workflow_instances" vào model User