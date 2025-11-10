# Tóm tắt API - Hệ thống Quản lý Tài liệu

## Tổng quan
Dự án có **29 API endpoints** được chia thành 2 nhóm chính:
- **API Test** (2 endpoints): `/test/*`
- **API Document** (27 endpoints): `/file/*`

---

## 1. API TEST (`/test`)

| Method | Endpoint | Chức năng |
|--------|----------|-----------|
| GET | `/test` | Test endpoint cơ bản |
| GET | `/test/categories` | Lấy danh sách tất cả categories |

---

## 2. API DOCUMENT (`/file`)

### 2.1. Upload & Xử lý File

| Method | Endpoint | Chức năng | UC |
|--------|----------|-----------|-----|
| POST | `/file/process_upload` | Xử lý file upload (Giai đoạn 1) | UC-39 |
| POST | `/file/insert` | Hoàn tất upload (Giai đoạn 2) | UC-39 |
| POST | `/file/photos/insert` | Thêm mô tả cho ảnh | - |

### 2.2. CRUD Document

| Method | Endpoint | Chức năng |
|--------|----------|-----------|
| GET | `/file/view` | Xem chi tiết tài liệu |
| PUT | `/file/edit` | Chỉnh sửa tài liệu |
| DELETE | `/file/delete` | Xóa tài liệu |
| POST | `/file/move_to_trash` | Chuyển vào thùng rác |
| GET | `/file/recover_files` | Khôi phục từ thùng rác |

### 2.3. Tìm kiếm & Lọc

| Method | Endpoint | Chức năng | UC |
|--------|----------|-----------|-----|
| GET | `/file/search` | Tìm kiếm tài liệu (NLP) | UC-73 |
| GET | `/file/search_photo` | Tìm kiếm ảnh theo mô tả | - |
| GET | `/file/get_by_category` | Lấy tài liệu theo category | - |
| GET | `/file/get_recent_files` | Lấy file gần đây | - |
| GET | `/file/get_recent_files_deleted` | Lấy file đã xóa gần đây | - |
| GET | `/file/get_document_by_metadata` | Tìm theo metadata | - |
| GET | `/file/get_document_by_orgnization` | Lấy theo organization | - |

### 2.4. Metadata

| Method | Endpoint | Chức năng | UC |
|--------|----------|-----------|-----|
| GET | `/file/suggest_metadata` | Gợi ý metadata | UC-73 |
| PUT | `/file/edit_metadata` | Chỉnh sửa metadata | - |
| PUT | `/file/edit_description_photo` | Chỉnh sửa mô tả ảnh | - |

### 2.5. Bảo mật & Mật khẩu

| Method | Endpoint | Chức năng |
|--------|----------|-----------|
| GET | `/file/set_password_document` | Đặt mật khẩu tài liệu |
| GET | `/file/remove_password_document` | Xóa mật khẩu tài liệu |
| GET | `/file/check_document_password` | Kiểm tra mật khẩu |
| GET | `/file/check_valid_document_password` | Kiểm tra có mật khẩu không |

### 2.6. So sánh & Kiểm tra Trùng lặp

| Method | Endpoint | Chức năng | UC |
|--------|----------|-----------|-----|
| GET | `/file/check_plagiarism` | Kiểm tra đạo văn | UC-88 |
| GET | `/file/compare_documents` | So sánh 2 tài liệu | - |
| POST | `/file/compare-images` | So sánh 2 ảnh | - |

### 2.7. Notification

| Method | Endpoint | Chức năng |
|--------|----------|-----------|
| POST | `/file/send_notification` | Gửi thông báo |

---

## 3. Luồng xử lý chính

### 3.1. Upload File (UC-39)

```
┌─────────────────────────────────────┐
│  1. POST /file/process_upload       │
│     - Validate file                 │
│     - Denoise (khử nhiễu)           │
│     - OCR                           │
│     - Kiểm tra trùng lặp (UC-88)    │
│     - Gợi ý metadata (UC-73)        │
│     - Watermark preview             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. POST /file/insert               │
│     - Nhúng watermark               │
│     - Upload lên Google Drive       │
│     - Lưu vào database              │
│     - Tự động luân chuyển (UC-84)   │
│     - Thiết lập quyền (UC-85/86)    │
└─────────────────────────────────────┘
```

### 3.2. Tìm kiếm (UC-73)

```
GET /file/search
├── Vectorize query (RoBERTa)
├── Tìm kiếm Elasticsearch
├── Tính relevance score
└── Trả về kết quả
```

### 3.3. Kiểm tra Trùng lặp (UC-88)

```
POST /file/process_upload
└── Kiểm tra trùng lặp
    ├── So sánh nội dung OCR
    ├── Tính similarity (>30% = trùng)
    └── Trả về 409 nếu trùng
```

---

## 4. Các Use Case (UC) được triển khai

| UC | Mô tả | API liên quan |
|----|-------|---------------|
| UC-39 | Upload và xử lý file | `/file/process_upload`, `/file/insert` |
| UC-73 | Gợi ý metadata và tìm kiếm | `/file/suggest_metadata`, `/file/search` |
| UC-84 | Tự động luân chuyển | `/file/insert` (auto-route) |
| UC-85 | Quản lý quyền truy cập (paid) | `/file/insert` (access_type) |
| UC-86 | Quản lý quyền truy cập (public) | `/file/insert` (public_link) |
| UC-87 | OCR và đếm trang | `/file/process_upload` (OCR) |
| UC-88 | Kiểm tra trùng lặp | `/file/process_upload`, `/file/check_plagiarism` |

---

## 5. Công nghệ sử dụng

- **Backend:** FastAPI
- **Database:** PostgreSQL
- **Search:** Elasticsearch
- **ML:** RoBERTa (Hugging Face)
- **OCR:** Google Vision API
- **Storage:** Google Drive
- **Image Processing:** OpenCV, PIL
- **Auth:** JWT

---

## 6. Quyền truy cập

### 6.1. RBAC
- `documents:upload`
- `documents:create`
- `documents:edit`
- `documents:delete`

### 6.2. User Roles
- `is_organization_manager`
- `is_dept_manager`
- Regular user

---

## 7. Giới hạn và Ràng buộc

- **Kích thước file:** Tối đa 50MB
- **Định dạng hỗ trợ:** PDF, DOCX, JPEG, PNG, TIFF, MP4, MP3, AVI, WAV
- **Ngưỡng trùng lặp:** 30%
- **Authentication:** Hiện tại hardcode user_id = 1

---

## 8. Error Codes

| Code | Mô tả |
|------|-------|
| 200 | Thành công |
| 400 | Bad Request (file quá lớn, định dạng không hỗ trợ) |
| 403 | Forbidden (không có quyền) |
| 409 | Conflict (trùng lặp) |
| 500 | Internal Server Error |

---

## 9. Scheduler Tasks

- **Xóa tài liệu cũ:** Chạy định kỳ
- **Xử lý file upload:** Background processing

---

## 10. Tài liệu chi tiết

Xem file `API_DOCUMENTATION.md` để biết chi tiết về từng API endpoint, request/response format, và luồng xử lý cụ thể.



