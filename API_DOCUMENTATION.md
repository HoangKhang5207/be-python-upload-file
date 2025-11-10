# Tài liệu API - Hệ thống Quản lý Tài liệu

## Tổng quan
Dự án này là một hệ thống quản lý tài liệu được xây dựng bằng FastAPI với các chức năng chính:
- Upload và xử lý file (PDF, DOCX, hình ảnh, video, audio)
- OCR và nhận dạng văn bản
- Kiểm tra trùng lặp tài liệu
- Quản lý metadata và phân loại
- Tìm kiếm tài liệu với Elasticsearch
- Quản lý quyền truy cập và bảo mật
- Watermark và bảo vệ tài liệu

## Cấu trúc API

### Base URL
- API Prefix: `/` (theo config.API_PREFIX)
- Documentation: `/docs` (Swagger UI)
- ReDoc: `/re-docs`

---

## 1. API TEST (Prefix: `/test`)

### 1.1. GET `/test`
**Chức năng:** Test endpoint cơ bản
**Luồng thực thi:**
1. Nhận request GET
2. Trả về chuỗi 'test'

**Response:**
```json
"test"
```

### 1.2. GET `/test/categories`
**Chức năng:** Lấy danh sách tất cả các danh mục
**Luồng thực thi:**
1. Gọi service `get_all_categories()`
2. Query tất cả categories từ database
3. Trả về danh sách categories dưới dạng dictionary

**Response:**
```json
[
  {
    "id": 1,
    "name": "Category Name",
    ...
  }
]
```

---

## 2. API DOCUMENT (Prefix: `/file`)

### 2.1. POST `/file/process_upload`
**Chức năng:** Xử lý file upload - Giai đoạn 1 (UC-39)
**Tags:** Document
**Luồng thực thi:**
1. **Kiểm tra quyền:** Lấy user_id từ request (hiện tại hardcode = 1)
2. **Kiểm tra file:**
   - Kiểm tra kích thước file (tối đa 50MB)
   - Kiểm tra định dạng file (PDF, DOCX, JPEG, PNG, TIFF, MP4, MP3, AVI, WAV)
3. **Khử nhiễu (Denoise):** Sử dụng OpenCV để khử nhiễu cho ảnh
4. **OCR:** Thực hiện OCR để trích xuất văn bản từ file
5. **Kiểm tra trùng lặp (UC-88):**
   - So sánh nội dung với các tài liệu hiện có
   - Ngưỡng trùng lặp: 30%
   - Nếu trùng lặp → trả về lỗi 409
6. **Gợi ý metadata (UC-73):** Tự động gợi ý metadata từ nội dung OCR
7. **Kiểm tra mâu thuẫn dữ liệu:** Kiểm tra xung đột trong metadata
8. **Watermark preview:** Tạo watermark preview để hiển thị

**Request:**
- `upload_file`: File (multipart/form-data)
- `duplicate_check_enabled`: boolean (default: true)

**Response (200):**
```json
{
  "status": "200",
  "message": "Xử lý file thành công. Vui lòng xem lại thông tin.",
  "data": {
    "ocrContent": "...",
    "total_pages": 10,
    "suggestedMetadata": {...},
    "warnings": [],
    "conflicts": [],
    "denoiseInfo": {...},
    "watermarkInfo": {...}
  }
}
```

**Response (409 - Trùng lặp):**
```json
{
  "status": "409",
  "message": "Phát hiện trùng lặp!",
  "duplicateData": {...}
}
```

---

### 2.2. POST `/file/insert`
**Chức năng:** Hoàn tất upload tài liệu - Giai đoạn 2 (UC-39)
**Tags:** Document
**Luồng thực thi:**
1. **Kiểm tra quyền:**
   - Kiểm tra user có quyền `documents:create`
   - Kiểm tra quyền truy cập category (ABAC)
2. **Kiểm tra file:** Validate kích thước và định dạng
3. **Kiểm tra trùng lặp tên file:** Đảm bảo tên file unique trong category
4. **Xử lý dung lượng:** Tính toán dung lượng lưu trữ
5. **Nhúng watermark:** Nhúng watermark vào file với mức độ bảo mật
6. **Upload lên Google Drive:** Upload file đã có watermark
7. **Lưu vào database:**
   - Tạo bản ghi Document với status="DRAFT", version="1.0"
   - Lưu metadata, OCR content, số trang
8. **Tự động luân chuyển (UC-84):** Kích hoạt workflow nếu cần
9. **Thiết lập quyền truy cập (UC-85/86):**
   - Nếu public: Tạo public link
   - Nếu paid: Đánh dấu is_paid=True

**Request:**
- `upload_file`: File
- `title`: string (required)
- `category_id`: int (required)
- `tags`: string (optional)
- `access_type`: string (required) - "private", "public", "paid"
- `confidentiality`: string (required) - "PUBLIC", "INTERNAL", "LOCKED"
- `description`: string (optional)
- `ocr_content`: string (optional)
- `total_pages`: int (default: 1)
- `key_values_json`: string (default: "{}")
- `summary`: string (optional)

**Response (200):**
```json
{
  "status": "200",
  "message": "Hoàn tất thành công!",
  "document": {
    "doc_id": 123,
    "version": "1.0",
    "status": "DRAFT" | "PROCESSING_WORKFLOW",
    "public_link": "...",
    ...
  },
  "autoRouteInfo": {...}
}
```

---

### 2.3. POST `/file/send_notification`
**Chức năng:** Gửi thông báo
**Luồng thực thi:**
1. Tạo request với headers (API key, Authorization token)
2. Gửi POST request đến service thông báo
3. Trả về kết quả

**Request:**
- `title`: string
- `content`: string
- `user_id`: int
- `doc_id`: int

**Response:**
```json
{
  "status": "200",
  "message": "Notification sent successfully"
}
```

---

### 2.4. PUT `/file/edit`
**Chức năng:** Chỉnh sửa tài liệu
**Luồng thực thi:**
1. **Kiểm tra quyền:** Kiểm tra user có quyền chỉnh sửa
2. **Kiểm tra category:** Kiểm tra quyền truy cập category
3. **Cập nhật:**
   - Cập nhật description nếu có
   - Cập nhật category_id nếu có
   - Cập nhật category trong Elasticsearch
4. **Lưu vào database**

**Request:**
- `doc_id`: int
- `description`: string (optional)
- `category_id`: int (optional)

**Response:**
```json
{
  "status": "200",
  "message": "Document updated successfully",
  "data": {...}
}
```

---

### 2.5. DELETE `/file/delete`
**Chức năng:** Xóa tài liệu
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `delete_document(doc_id, user_id)`
3. Xóa tài liệu khỏi database
4. Trả về thông tin tài liệu đã xóa

**Request:**
- `doc_id`: int

**Response:**
```json
{
  "status": "200",
  "message": "Document deleted successfully",
  "data": {...}
}
```

---

### 2.6. POST `/file/move_to_trash`
**Chức năng:** Chuyển tài liệu vào thùng rác
**Luồng thực thi:**
1. **Kiểm tra quyền:** Kiểm tra user có quyền
2. **Kiểm tra category:** Kiểm tra quyền truy cập category
3. **Chuyển vào thùng rác:** Đánh dấu tài liệu là đã xóa (soft delete)

**Request:**
- `doc_id`: int

**Response:**
```json
{
  "status": "200",
  "message": "Document moved to trash successfully",
  "data": {...}
}
```

---

### 2.7. GET `/file/view`
**Chức năng:** Xem chi tiết tài liệu
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `view_document(doc_id, user_id)`
3. Kiểm tra quyền truy cập
4. Trả về thông tin tài liệu

**Request:**
- `doc_id`: int

**Response:**
```json
{
  "status": "200",
  "message": "Document retrieved successfully",
  "data": {...}
}
```

---

### 2.8. GET `/file/get_by_category`
**Chức năng:** Lấy danh sách tài liệu theo category
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `get_documents_by_category_service(category_id, page, page_size, user_id)`
3. Áp dụng phân quyền (chỉ lấy tài liệu user có quyền xem)
4. Trả về danh sách với phân trang

**Request:**
- `category_id`: int
- `page`: int (default: 1)
- `page_size`: int (default: 10)

**Response:**
```json
{
  "status": "200",
  "message": "Documents retrieved successfully",
  "data": [...],
  "page": 1,
  "total_pages": 10
}
```

---

### 2.9. GET `/file/get_recent_files`
**Chức năng:** Lấy danh sách file gần đây
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `get_recent_documents_service(time, user_id, page, page_size)`
3. Lọc theo thời gian và user
4. Trả về danh sách với phân trang

**Request:**
- `time`: date
- `page`: int (default: 1)
- `page_size`: int (default: 30)

**Response:**
```json
{
  "status": "200",
  "message": "Documents retrieved successfully",
  "data": [...],
  "page": 1,
  "total_pages": 5
}
```

---

### 2.10. GET `/file/get_recent_files_deleted`
**Chức năng:** Lấy danh sách file đã xóa gần đây
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `get_recent_documents_deleted_service(time, user_id, page, page_size)`
3. Lọc các tài liệu đã xóa (soft delete)
4. Trả về danh sách với phân trang

**Request:**
- `time`: date
- `page`: int (default: 1)
- `page_size`: int (default: 10)

**Response:**
```json
{
  "status": "200",
  "message": "Documents retrieved successfully",
  "data": [...],
  "page": 1,
  "total_pages": 3
}
```

---

### 2.11. GET `/file/recover_files`
**Chức năng:** Khôi phục file từ thùng rác
**Luồng thực thi:**
1. **Kiểm tra quyền:** Kiểm tra user có quyền
2. **Kiểm tra category:** Kiểm tra quyền truy cập category
3. **Khôi phục:** Gọi service `recover_files(doc_id, user_id)`
4. Trả về thông tin tài liệu đã khôi phục

**Request:**
- `doc_id`: int

**Response:**
```json
{
  "status": "200",
  "message": "Document recovered successfully",
  "data": {...}
}
```

---

### 2.12. GET `/file/search`
**Chức năng:** Tìm kiếm tài liệu với NLP (UC-73)
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `search_documents(question, client, category_id, type_doc, start, end, user_id)`
3. **Xử lý tìm kiếm:**
   - Sử dụng Elasticsearch
   - Sử dụng model embedding (RoBERTa) để vectorize query
   - Tìm kiếm semantic similarity
   - Tính relevance score
4. **Lọc và sắp xếp:**
   - Lọc các tài liệu có relevance_score
   - Sắp xếp theo relevance_score giảm dần
5. Trả về kết quả với phân trang

**Request:**
- `category_id`: string (optional)
- `type_doc`: string (optional)
- `question`: string (optional, default: "")
- `page_num`: int (default: 1)
- `page_size`: int (default: 10)

**Response:**
```json
{
  "status": 200,
  "data": [
    {
      "id": 123,
      "relevance_score": 0.95,
      ...
    }
  ],
  "total": 50,
  "normalized_question": "...",
  "currentPage": 1,
  "totalPage": 5
}
```

---

### 2.13. GET `/file/suggest_metadata`
**Chức năng:** Gợi ý metadata từ các tài liệu hiện có
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `filter_metadata_by_user(user_id, ...)`
3. **Lọc metadata:**
   - Lọc theo các tham số: document_number, issuing_authority, date_of_issuance, signature, agency_address
   - Chỉ lấy metadata từ tài liệu của user
4. **Nhóm dữ liệu:** Nhóm các giá trị unique theo từng trường
5. Trả về danh sách các giá trị gợi ý

**Request:**
- `document_number`: string (optional)
- `issuing_authority`: string (optional)
- `date_of_issuance`: string (optional)
- `signature`: string (optional)
- `agency_address`: string (optional)

**Response:**
```json
{
  "status": 200,
  "data": {
    "document_number": ["DOC001", "DOC002"],
    "issuing_authority": ["Cơ quan A", "Cơ quan B"],
    ...
  }
}
```

---

### 2.14. GET `/file/get_document_by_metadata`
**Chức năng:** Tìm tài liệu theo metadata
**Luồng thực thi:**
1. Lấy user_id từ request
2. Kiểm tra nếu tất cả tham số đều None → trả về danh sách rỗng
3. Gọi service `get_document_by_meta(user_id, ...)`
4. Tìm kiếm tài liệu theo các trường metadata
5. Trả về danh sách tài liệu khớp

**Request:**
- `document_number`: string (optional)
- `issuing_authority`: string (optional)
- `date_of_issuance`: string (optional)
- `signature`: string (optional)
- `agency_address`: string (optional)

**Response:**
```json
{
  "status": 200,
  "data": [...]
}
```

---

### 2.15. GET `/file/set_password_document`
**Chức năng:** Đặt mật khẩu cho tài liệu
**Luồng thực thi:**
1. **Kiểm tra quyền:** Kiểm tra user có quyền
2. **Kiểm tra category:** Kiểm tra quyền truy cập category
3. Gọi service `set_password_document(doc_id, password)`
4. Mã hóa và lưu mật khẩu vào database

**Request:**
- `doc_id`: int
- `password`: string

**Response:**
```json
{
  "status": "200",
  "message": "Document password set successfully"
}
```

---

### 2.16. GET `/file/remove_password_document`
**Chức năng:** Xóa mật khẩu của tài liệu
**Luồng thực thi:**
1. **Kiểm tra quyền:** Kiểm tra user có quyền
2. **Kiểm tra category:** Kiểm tra quyền truy cập category
3. Gọi service `remove_password_document(doc_id)`
4. Xóa mật khẩu khỏi database

**Request:**
- `doc_id`: int

**Response:**
```json
{
  "status": "200",
  "message": "Document password removed successfully"
}
```

---

### 2.17. GET `/file/check_document_password`
**Chức năng:** Kiểm tra mật khẩu tài liệu
**Luồng thực thi:**
1. Gọi service `check_document_password(doc_id, password)`
2. So sánh mật khẩu đã hash
3. Trả về kết quả kiểm tra

**Request:**
- `doc_id`: int
- `password`: string

**Response:**
```json
{
  "status": "200",
  "message": "Document password checked successfully",
  "data": {...}
}
```

---

### 2.18. GET `/file/check_valid_document_password`
**Chức năng:** Kiểm tra tài liệu có mật khẩu không
**Luồng thực thi:**
1. Gọi service `check_valid_document_password(doc_id)`
2. Kiểm tra xem tài liệu có mật khẩu hay không
3. Trả về kết quả

**Request:**
- `doc_id`: int

**Response:**
```json
{
  "status": "200",
  "message": "Document password checked successfully",
  "data": {
    "has_password": true
  }
}
```

---

### 2.19. GET `/file/get_document_by_orgnization`
**Chức năng:** Lấy danh sách tài liệu theo tổ chức
**Luồng thực thi:**
1. Gọi service `get_documents_by_organization_service(organization_id)`
2. Lấy tất cả tài liệu thuộc organization
3. Trả về danh sách document IDs

**Request:**
- `organization_id`: int

**Response:**
```json
{
  "status": 200,
  "data": [1, 2, 3, ...]
}
```

---

### 2.20. POST `/file/photos/insert`
**Chức năng:** Thêm mô tả cho ảnh
**Luồng thực thi:**
1. Đọc file content
2. Gọi service `generate_image_description(doc_id, file_content)`
3. Sử dụng Google Vision API để tạo mô tả ảnh
4. Lưu mô tả vào database
5. Trả về mô tả

**Request:**
- `doc_id`: int
- `file`: File (multipart/form-data)

**Response:**
```json
{
  "description": "Mô tả ảnh..."
}
```

---

### 2.21. GET `/file/search_photo`
**Chức năng:** Tìm kiếm ảnh theo mô tả
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `search_image_descriptions(query, user_id)`
3. Tìm kiếm ảnh có mô tả khớp với query
4. Trả về danh sách tài liệu ảnh

**Request:**
- `query`: string

**Response:**
```json
{
  "documents": [...]
}
```

---

### 2.22. GET `/file/check_plagiarism`
**Chức năng:** Kiểm tra đạo văn (UC-88)
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `check_plagiarism(doc_id)`
3. **Xử lý:**
   - Lấy nội dung tài liệu
   - So sánh với các tài liệu khác
   - Sử dụng thuật toán Rabin-Karp hoặc cosine similarity
   - Tính điểm tương đồng
4. Trả về kết quả kiểm tra

**Request:**
- `doc_id`: int

**Response:**
```json
{
  "status": "200",
  "data": {
    "similarity_score": 0.85,
    "similar_documents": [...]
  }
}
```

---

### 2.23. GET `/file/compare_documents`
**Chức năng:** So sánh hai tài liệu
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `compare_document_similarity(doc_id1, doc_id2)`
3. **Xử lý:**
   - Lấy nội dung hai tài liệu
   - Vectorize nội dung
   - Tính cosine similarity
   - Trả về điểm tương đồng

**Request:**
- `doc_id1`: int
- `doc_id2`: int

**Response:**
```json
{
  "status": "200",
  "data": {
    "similarity": 0.92
  }
}
```

---

### 2.24. POST `/file/compare-images`
**Chức năng:** So sánh hai ảnh
**Luồng thực thi:**
1. Lấy user_id từ request
2. Lấy ảnh từ database: `get_image_from_db(doc_id1)`, `get_image_from_db(doc_id2)`
3. **Xử lý:**
   - Decode base64 thành image
   - Resize ảnh để cùng kích thước (nếu cần)
   - Tính SSIM (Structural Similarity Index)
4. Trả về điểm tương đồng

**Request:**
- `doc_id1`: int
- `doc_id2`: int

**Response:**
```json
{
  "similarity": 0.95
}
```

---

### 2.25. PUT `/file/edit_metadata`
**Chức năng:** Chỉnh sửa metadata của tài liệu
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `update_metadata(document_id, ...)`
3. Cập nhật các trường metadata:
   - document_number
   - issuing_authority
   - date_of_issuance
   - signature
   - agency_address
4. Lưu vào database và Elasticsearch
5. Trả về metadata đã cập nhật

**Request:**
- `document_id`: int
- `document_number`: string (optional)
- `issuing_authority`: string (optional)
- `date_of_issuance`: string (optional)
- `signature`: string (optional)
- `agency_address`: string (optional)

**Response:**
```json
{
  "status": 200,
  "data": {...}
}
```

---

### 2.26. PUT `/file/edit_description_photo`
**Chức năng:** Chỉnh sửa mô tả ảnh
**Luồng thực thi:**
1. Lấy user_id từ request
2. Gọi service `update_photo_description(doc_id, new_description)`
3. Cập nhật mô tả ảnh trong database
4. Trả về mô tả đã cập nhật

**Request:**
- `doc_id`: int
- `new_description`: string

**Response:**
```json
{
  "status": 200,
  "data": {...}
}
```

---

## 3. Luồng xử lý chính

### 3.1. Luồng Upload File (UC-39)

```
1. POST /file/process_upload
   ├── Kiểm tra quyền
   ├── Validate file (kích thước, định dạng)
   ├── Denoise (khử nhiễu)
   ├── OCR (trích xuất văn bản)
   ├── Kiểm tra trùng lặp (UC-88)
   │   └── Nếu trùng → Trả về 409
   ├── Gợi ý metadata (UC-73)
   ├── Kiểm tra mâu thuẫn dữ liệu
   └── Watermark preview
   → Trả về dữ liệu để review

2. POST /file/insert
   ├── Kiểm tra quyền
   ├── Validate file
   ├── Kiểm tra trùng lặp tên file
   ├── Tính dung lượng
   ├── Nhúng watermark
   ├── Upload lên Google Drive
   ├── Lưu vào database (status=DRAFT, version=1.0)
   ├── Tự động luân chuyển (UC-84)
   ├── Thiết lập quyền truy cập (UC-85/86)
   └── Trả về kết quả
```

### 3.2. Luồng Tìm kiếm (UC-73)

```
GET /file/search
├── Lấy user_id từ request
├── Vectorize query (RoBERTa model)
├── Tìm kiếm trong Elasticsearch
│   ├── Semantic search
│   ├── Tính relevance score
│   └── Lọc theo category, type_doc
├── Sắp xếp theo relevance_score
└── Trả về kết quả với phân trang
```

### 3.3. Luồng Kiểm tra Trùng lặp (UC-88)

```
POST /file/process_upload (bước kiểm tra trùng lặp)
├── Lấy nội dung OCR
├── So sánh với các tài liệu trong organization
├── Tính similarity score (Rabin-Karp hoặc cosine similarity)
├── Nếu similarity > 30% → Trùng lặp
│   └── Trả về 409 với thông tin trùng lặp
└── Nếu không trùng → Tiếp tục xử lý
```

---

## 4. Các công nghệ sử dụng

- **Framework:** FastAPI
- **Database:** PostgreSQL
- **Search Engine:** Elasticsearch
- **ML Model:** RoBERTa (Hugging Face)
- **OCR:** Google Vision API, Tesseract
- **Storage:** Google Drive
- **Image Processing:** OpenCV, PIL
- **Document Processing:** PyPDF2, python-docx, openpyxl
- **Authentication:** JWT
- **Migration:** Alembic

---

## 5. Quyền và Phân quyền

### 5.1. RBAC (Role-Based Access Control)
- `documents:upload` - Quyền upload tài liệu
- `documents:create` - Quyền tạo tài liệu
- `documents:edit` - Quyền chỉnh sửa tài liệu
- `documents:delete` - Quyền xóa tài liệu

### 5.2. ABAC (Attribute-Based Access Control)
- Kiểm tra organization_id
- Kiểm tra department_id
- Kiểm tra category permissions

### 5.3. User Roles
- `is_organization_manager` - Quản lý tổ chức
- `is_dept_manager` - Quản lý phòng ban
- Regular user - Người dùng thường

---

## 6. Scheduler Tasks

### 6.1. Xóa tài liệu cũ
- Chạy định kỳ để xóa tài liệu đã hết hạn
- Function: `schedule_delete_old_documents()`

### 6.2. Xử lý file upload
- Xử lý các file upload trong background
- Function: `schedule_file_uploads_processing()`

---

## 7. Lưu ý

1. **Authentication:** Hiện tại middleware authentication đang bị comment, user_id được hardcode = 1
2. **File Size Limit:** Tối đa 50MB
3. **Duplicate Threshold:** 30% (có thể điều chỉnh)
4. **Supported Formats:** PDF, DOCX, JPEG, PNG, TIFF, MP4, MP3, AVI, WAV
5. **Watermark:** Hỗ trợ watermark cho ảnh và PDF
6. **OCR:** Hỗ trợ OCR cho PDF và ảnh
7. **Search:** Sử dụng semantic search với Elasticsearch và RoBERTa

---

## 8. Error Handling

### 8.1. HTTP Status Codes
- `200` - Thành công
- `400` - Bad Request (file quá lớn, định dạng không hỗ trợ)
- `403` - Forbidden (không có quyền)
- `409` - Conflict (trùng lặp)
- `500` - Internal Server Error

### 8.2. Error Response Format
```json
{
  "status": "400",
  "message": "Error message"
}
```

---

## 9. Testing

- Sử dụng Pytest cho unit testing
- Test files nằm trong thư mục `tests/`
- Có test cho login và register

---

## 10. Deployment

- Hỗ trợ Docker và Docker Compose
- Cấu hình CORS cho các domain cụ thể
- Sử dụng environment variables cho configuration
- Logging được cấu hình trong `logging.ini`



