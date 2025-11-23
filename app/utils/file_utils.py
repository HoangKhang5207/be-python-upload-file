"""
File utilities for document processing
"""
import io
import logging

from app.services.document_service import get_mime_type

# Try to import pdfplumber, fallback to PyPDF2 if not available
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    import PyPDF2
    HAS_PDFPLUMBER = False
    logging.warning("pdfplumber not available, using PyPDF2 fallback")


# Constants from api_document.py
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_MIME_TYPES = [
    'application/pdf', 
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document', # .docx
    'image/jpeg', 
    'image/png', 
    'image/tiff', 
    'video/mp4', 
    'audio/mpeg', # .mp3
    'video/x-msvideo', # .avi
    'audio/wav' # .wav
]
DUPLICATE_THRESHOLD = 0.30

# =========================================================================
# HÀM KIỂM TRA KÍCH THƯỚC FILE (File Size Check)
# =========================================================================
def is_file_size_exceeded(file_size_bytes: int) -> bool:
    """
    Kiểm tra xem kích thước file có vượt quá giới hạn MAX_FILE_SIZE_BYTES không.
    
    Args:
        file_size_bytes (int): Kích thước file (bytes)
    
    Returns:
        bool: True nếu vượt quá, False nếu hợp lệ
    """
    return file_size_bytes > MAX_FILE_SIZE_BYTES

# =========================================================================
# HÀM KIỂM TRA ĐỊNH DẠNG FILE (MIME Type Validation)
# =========================================================================
def is_mime_type_allowed(mime_type: str) -> bool:
    """
    Kiểm tra xem MIME type có nằm trong danh sách ALLOWED_MIME_TYPES không.
    
    Args:
        mime_type (str): MIME type cần kiểm tra
    
    Returns:
        bool: True nếu được phép, False nếu không
    """
    return mime_type in ALLOWED_MIME_TYPES

# =========================================================================
# HÀM PHÂN LOẠI LOẠI FILE (File Type Classification)
# =========================================================================
def classify_file_type(file_content: bytes, mime_type: str) -> str:
    """
    Phân loại loại file dựa trên MIME type và nội dung file.
    
    Args:
        file_content (bytes): Nội dung file dưới dạng bytes
        mime_type (str): MIME type của file (từ ALLOWED_MIME_TYPES)
    
    Returns:
        str: Loại file - "image", "text", "audio", "video"
    
    Mapping:
        - image/jpeg, image/png, image/tiff → "image"
        - application/vnd.openxmlformats-officedocument.wordprocessingml.document → "text"
        - application/pdf:
            - Nếu có text → "text"
            - Nếu không có text (PDF scan từ ảnh) → "image"
        - audio/mpeg, audio/wav → "audio"
        - video/mp4, video/x-msvideo → "video"
    """
    
    # Phân loại sơ bộ theo MIME type
    if mime_type in ['image/jpeg', 'image/png', 'image/tiff']:
        return "image"
    
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return "text"
    
    elif mime_type in ['audio/mpeg', 'audio/wav']:
        return "audio"
    
    elif mime_type in ['video/mp4', 'video/x-msvideo']:
        return "video"
    
    elif mime_type == 'application/pdf':
        # Xử lý riêng cho PDF: kiểm tra xem có text hay không
        try:
            pdf_file = io.BytesIO(file_content)
            total_text = ""
            
            if HAS_PDFPLUMBER:
                # Dùng pdfplumber nếu có (chính xác hơn)
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            total_text += page_text
            else:
                # Fallback sang PyPDF2
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        total_text += page_text
            
            # Kiểm tra số ký tự thực tế (loại bỏ whitespace)
            text_length = len(total_text.strip())
            
            # Nếu text rất ít (< 100 ký tự) → coi là PDF scan (image)
            # Tăng ngưỡng để bắt được PDF scan tốt hơn
            if text_length < 100:
                return "image"  # PDF scan từ ảnh
            else:
                return "text"   # PDF có text thật
        except Exception as e:
            # Nếu có lỗi khi đọc PDF, mặc định coi là text
            logging.warning(f"Lỗi khi phân loại PDF: {str(e)}")
            return "text"
    
    # Fallback (không nên xảy ra nếu MIME type hợp lệ)
    return "text"
