import base64
import traceback
import logging

import cv2
import requests
from typing import Union
from fastapi import APIRouter, UploadFile, Body, Request, HTTPException, File
from pydantic import BaseModel
from datetime import date
import time
import mimetypes
import json
from fastapi.responses import JSONResponse
from transformers import RobertaModel
from app.core.config import settings
from app.db.base import SessionLocal
from app.db.engine_default import session_scope

from app.services import document_service
from elasticsearch import Elasticsearch
from app.services.document_service import (insert_document, update_category_in_elasticsearch, document_to_dict,
                                           insert_file_upload, generate_image_description, search_image_descriptions,
                                           serialize_document, check_plagiarism, compare_document_similarity,
                                           get_image_from_db, create_image_from_base64, calculate_ssim,
                                           get_user_by_id, get_category_by_id, check_duplicate_document, process_file,
                                           
                                        get_mime_type,
                                        denoise_image,
                                        perform_ocr,
                                        check_duplicates_by_content,
                                        suggest_metadata_from_content,
                                        check_data_conflicts,
                                        embed_watermark_preview,
                                        
                                        embed_watermark_final,
                                        upload_file_to_drive, # (Hàm đã được cập nhật)
                                        trigger_auto_route,
                                        save_public_link
                                           )

router = APIRouter()
model = settings.MODEL
model_embeding = RobertaModel.from_pretrained(model)
elasticsearch_url = settings.ELASTICSEARCH_ENPOINT
client = Elasticsearch([elasticsearch_url])
logging.basicConfig(level=logging.DEBUG)
# photoprism_url = settings.PHOTOPRISM_URL
# admin_password = settings.PHOTOPRISM_ADMIN_PASSWORD
# google_application_credentials = settings.GOOGLE_APPLICATION_CREDENTIALS

SCOPES = ['https://www.googleapis.com/auth/drive']

DEFAULT_TEST_USER_ID = 1


def get_request_user_id(request: Request, default_user_id: int = DEFAULT_TEST_USER_ID) -> int:
    """
    Lấy user_id từ request scope. Khi middleware auth bị tắt hoặc thiếu token,
    fallback về user_id mặc định để tránh lỗi 500.
    """
    user_id = request.scope.get('user_id')
    if user_id is None:
        logging.warning("user_id is missing from request scope; falling back to default user_id=%s", default_user_id)
        return default_user_id
    return user_id


# credentials = service_account.Credentials.from_service_account_file(
#     google_application_credentials, scopes=SCOPES
# )
#drive_service = build('drive', 'v3', credentials=credentials)

# (Các hằng số cho UC-39)
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
# Định dạng file được phép theo UC-39
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
# Ngưỡng trùng lặp (UC-39/UC-88)
DUPLICATE_THRESHOLD = 0.30

# =========================================================================
# API GIAI ĐOẠN 1: XỬ LÝ (Processing)
# (Khớp với hàm `processFile` của frontend UC39_UploadPage.jsx)
# =========================================================================

@router.post("/process_upload", tags=["Document"])
async def process_upload_file(
    request: Request,
    upload_file: UploadFile = File(...),
    duplicate_check_enabled: bool = Body(True)
):
    """
    API Giai đoạn 1 (UC-39):
    Thực hiện các bước xử lý file (Denoise, OCR, Check Trùng lặp, Gợi ý Metadata)
    và trả về JSON cho frontend review (Step 3).
    """
    try:
        # Bước 1: Kiểm tra quyền (RBAC: documents:upload)
        # (Sử dụng user_id từ middleware, tạm hardcode)
        user_id = get_request_user_id(request)
        user = document_service.get_user_by_id(user_id)
        if not user:
             raise HTTPException(status_code=403, detail="Không tìm thấy người dùng.")
        
        # (Giả lập kiểm tra RBAC)
        # if not user.can('documents:upload'):
        #     raise HTTPException(status_code=403, detail="403: Insufficient permissions (documents:upload)")

        # Đọc nội dung file
        file_content = await upload_file.read()
        await upload_file.seek(0) # Reset con trỏ file

        # Bước 2: Kiểm tra định dạng và kích thước (UC-39)
        if len(file_content) > MAX_FILE_SIZE_BYTES:
             raise HTTPException(status_code=400, detail=f"400: Kích thước file vượt quá {MAX_FILE_SIZE_MB}MB.")
        
        file_mime_type = document_service.get_mime_type(file_content)
        if file_mime_type not in ALLOWED_MIME_TYPES:
             raise HTTPException(status_code=400, detail=f"400: Định dạng file {file_mime_type} không được hỗ trợ.")

        # --- Bắt đầu luồng xử lý (Step 2 - 4) ---

        # Bước 3: Khử nhiễu (UC-39 Denoise)
        denoise_result = await document_service.denoise_image(file_content, upload_file.filename)

        # Bước 4: OCR & Đếm trang (UC-87)
        # (Sử dụng content đã khử nhiễu)
        ocr_result = await document_service.perform_ocr(denoise_result['content'], upload_file.filename)
        ocr_content = ocr_result['ocrContent']
        total_pages = ocr_result['total_pages'] # Lấy số trang

        # Bước 5: Kiểm tra trùng lặp (UC-88)
        if duplicate_check_enabled:
            duplicate_result = await document_service.check_duplicates_by_content(
                ocr_content=ocr_content, 
                organization_id=user.organization_id,
                user_id=user.id,
                similarity_threshold=DUPLICATE_THRESHOLD
            )
            
            if duplicate_result['isDuplicate']:
                # Frontend (UC39_UploadPage.jsx) chờ một lỗi 409
                return JSONResponse(status_code=409, content={
                    "status": "409", 
                    "message": "Phát hiện trùng lặp!",
                    "duplicateData": duplicate_result['duplicateData'] # Gửi dữ liệu trùng lặp về
                })

        # Bước 6: Gợi ý Metadata (UC-73)
        metadata_result = await document_service.suggest_metadata_from_content(
            ocr_content, 
            upload_file.filename
        )

        # Bước 7: Kiểm tra mâu thuẫn dữ liệu (UC-39/UC-73)
        conflict_result = document_service.check_data_conflicts(
            metadata_result['suggestedMetadata'].get('key_values', {})
        )

        # Bước 8: Giả lập Watermark (Chỉ để hiển thị)
        watermark_result = await document_service.embed_watermark_preview(
            denoise_result['content'], 
            upload_file.filename
        )

        # --- Kết hợp kết quả trả về ---
        full_api_response = {
            "ocrContent": ocr_content,
            "total_pages": total_pages, # <-- Trả về số trang
            "suggestedMetadata": metadata_result['suggestedMetadata'],
            "warnings": metadata_result.get('warnings', []),
            "conflicts": conflict_result.get('conflicts', []),
            "denoiseInfo": denoise_result.get('denoiseInfo', {}),
            "watermarkInfo": watermark_result.get('watermarkInfo', {}),
        }

        # Trả về 200 OK với đầy đủ dữ liệu
        return JSONResponse(status_code=200, content={
            "status": "200",
            "message": "Xử lý file thành công. Vui lòng xem lại thông tin.",
            "data": full_api_response
        })

    except HTTPException as e:
        # Bắt lại lỗi HTTP (ví dụ: 400, 403, 409)
        logging.error(f"Lỗi HTTP trong Giai đoạn 1 (/process_upload): {e.detail}")
        if e.status_code == 409:
             return JSONResponse(status_code=409, content=e.detail) # Đảm bảo lỗi 409 được trả về
        return JSONResponse(status_code=e.status_code, content={"status": str(e.status_code), "message": e.detail})
        
    except Exception as e:
        error_info = traceback.format_exc()
        logging.error(f"Lỗi máy chủ Giai đoạn 1 (/process_upload): {error_info}")
        return JSONResponse(status_code=500, content={"status": "500",
                                                      "message": "Đã xảy ra lỗi trong quá trình xử lý file: " + str(e)})

# =========================================================================
# API GIAI ĐOẠN 2: HOÀN TẤT (Finalize)
# (Khớp với hàm `handleFinalize` của frontend UC39_UploadPage.jsx)
# =========================================================================

@router.post("/insert", tags=["Document"])
async def finalize_document_upload( # Đổi tên hàm
    request: Request, 
    
    # --- Dữ liệu file ---
    upload_file: UploadFile = File(...),
    
    # --- Dữ liệu metadata từ Form (Step 3) ---
    title: str = Body(...),
    category_id: int = Body(...),
    tags: str = Body(None), # Frontend gửi "tag1, tag2"
    access_type: str = Body(...), # Frontend gửi "private", "public", "paid"
    confidentiality: str = Body(...), # Frontend gửi "PUBLIC", "INTERNAL", "LOCKED"
    
    # --- Dữ liệu ẩn (đã xử lý ở Giai đoạn 1) ---
    ocr_content: str = Body(None), # Nội dung text (nếu có)
    total_pages: int = Body(1), # Số trang (đã đếm)
    key_values_json: str = Body("{}"), # JSON string của key_values
    summary: str = Body(None), # Tóm tắt (nếu có)
    description: str = Body(None) # (Giữ lại trường description gốc)
):
    """
    API Giai đoạn 2 (UC-39):
    Thực hiện các bước cuối cùng (Watermark, Lưu trữ, Auto-Route)
    sau khi người dùng đã xác nhận metadata.
    """
    try:
        # Bước 1: Kiểm tra quyền (RBAC: documents:create)
        user_id = get_request_user_id(request)
        user = document_service.get_user_by_id(user_id)
        
        # (Kiểm tra RBAC và ABAC cho danh mục)
        if not user.is_organization_manager and not user.is_dept_manager:
            return JSONResponse(status_code=403,
                                content={"status": "403", "message": "Bạn không có quyền thêm tài liệu (thiếu 'documents:create')"})
        
        category = document_service.get_category_by_id(category_id)
        if category.organization_id != user.organization_id:
            return JSONResponse(status_code=403, content={"status": "403",
                                                          "message": "Bạn không có quyền thêm tài liệu vào danh mục này"})
        elif not user.is_organization_manager and category.department_id != user.dept_id:
            return JSONResponse(status_code=403, content={"status": "403",
                                                          "message": "Bạn không có quyền thêm tài liệu vào danh mục này"})
        
        # Kiểm tra định dạng file và kích thước (≤50MB)
        upload_file.file.seek(0, 2)
        file_size = upload_file.file.tell()
        upload_file.file.seek(0)
        if file_size > MAX_FILE_SIZE_BYTES:
             return JSONResponse(status_code=400, content={"status": "400", "message": "Lỗi: Kích thước file vượt quá 50MB."})
        
        allowed_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'image/jpeg', 'image/png', 'image/tiff', 'video/mp4', 'audio/mpeg', 'video/x-msvideo', 'audio/wav']
        file_mime_type, _ = mimetypes.guess_type(upload_file.filename)
        
        if file_mime_type not in allowed_types:
             return JSONResponse(status_code=400, content={"status": "400", "message": f"Định dạng file {file_mime_type} không được hỗ trợ."})

        await upload_file.seek(0)

        # Kiểm tra trùng lặp TÊN FILE (vẫn nên giữ)
        unique_filename = document_service.check_duplicate_document(upload_file.filename, category_id)
        
        # Kiểm tra dung lượng lưu trữ (vẫn giữ)
        storage_capacity, storage_unit = document_service.process_file(upload_file, user_id)
        
        # Bước 3.2: Nhúng Watermark (UC-39)
        file_content = await upload_file.read()
        final_file_content = await document_service.embed_watermark_final(
            file_content, 
            upload_file.filename,
            confidentiality
        )

        # Upload file đã có watermark
        file_id, file_path, local_file_path = document_service.upload_file_to_drive(
            final_file_content, 
            unique_filename
        )
        
        # Xử lý Photo ID cho ảnh
        file_extension = upload_file.filename.split('.')[-1].lower()
        allowed_image_extensions = ['png', 'jpg', 'jpeg', 'webp']
        photo_id = None
        if file_extension in allowed_image_extensions:
            photo_id = base64.b64encode(final_file_content).decode('utf-8')

        # --- Xây dựng metadata object đầy đủ ---
        final_metadata = {
            "title": title,
            "category_id": category_id,
            "tags": tags.split(',') if tags else [],
            "access_type": access_type,
            "confidentiality": confidentiality,
            "description": description,
            "ocrContent": ocr_content,
            "summary": summary,
            "key_values": json.loads(key_values_json)
        }

        # Bước 5: Lưu trữ và tạo bản ghi (status=DRAFT, version=1.0)
        inserted_document = await document_service.insert_document(
            filename=unique_filename,
            user=user,
            category_id=category_id,
            description=description, # (Trường description gốc)
            file_type=file_extension,
            file_id=file_id,
            file_path=file_path,
            storage_capacity=storage_capacity,
            storage_unit=storage_unit,
            photo_id=photo_id,
            # --- Các trường mới từ UC-39/Model ---
            final_metadata=final_metadata,
            status_str="DRAFT", # UC-39 yêu cầu
            version="1.0",  # UC-39 yêu cầu
            total_pages=total_pages # Đã lấy từ Giai đoạn 1
        )
        
        session = SessionLocal()
        inserted_document = session.merge(inserted_document)
        session.close()

        # (Lưu file upload vào DB - cho background processing nếu cần)
        # (Hàm này có thể bị xóa nếu file đã xử lý xong)
        await insert_file_upload(inserted_document.id, user_id, local_file_path)

        # Bước 6: Tự động luân chuyển (UC-84)
        auto_route_info = await document_service.trigger_auto_route(
            inserted_document, 
            final_metadata,
            user
        )

        # Bước 7: Thiết lập quyền truy cập (UC-85/86)
        public_link = None
        if access_type == 'public':
            public_link = await document_service.save_public_link(inserted_document.id, user_id)
        # (Logic 'paid' (UC-85) đã được xử lý bởi cờ `is_paid=True` trong `insert_document`)

        # Bước 8: Hiển thị kết quả (Khớp với Step4_Result)
        # (Sử dụng hàm serialize_document để lấy thông tin đầy đủ)
        serialized_document = document_service.serialize_document(inserted_document)
        
        # Cập nhật các trường trả về (model gốc không có, nên ta tự thêm)
        serialized_document.update({
            "doc_id": inserted_document.id,
            "version": inserted_document.version,
            "status": auto_route_info.get("triggered", False)
                      and "PROCESSING_WORKFLOW" 
                      or "DRAFT", # Phản ánh đúng trạng thái sau auto-route
            "public_link": public_link
        })

        return JSONResponse(status_code=200, content={
            "status": "200", 
            "message": "Hoàn tất thành công!",
            "document": serialized_document,
            "autoRouteInfo": auto_route_info # Trả về thông tin auto-route
        })
        
    except Exception as e:
        error_info = traceback.format_exc()
        logging.error(f"Lỗi máy chủ Giai đoạn 2 (/insert): {error_info}")
        return JSONResponse(status_code=500, content={"status": "500",
                                                      "message": "Đã xảy ra lỗi trong quá trình hoàn tất tài liệu: " + str(e)})

@router.post("/send_notification")
async def send_notification(title: str, content: str, user_id: int, doc_id: int):
    try:
        headers = {
            "x-internal-api-key": "Ic0ilkVIg6nXRROAX7ytfs4zC9yfdM6Fhbgmr1bbCCmKIFnHPSXSgT4l2W58htdfhr7HyQJKRTiyycHLLhIJBsuEcLnNbOMsfD99",
            "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJ1c2VyX2NyZWRlbnRpYWxzIiwiZW1haWwiOiJhZG1pbkBnbWFpbC5jb20iLCJleHAiOjE3MjYyNDE4NDMsImlhdCI6MTcyMzY0OTg0MywiaXNzIjoiR2VuaUZhc3QtU2VhcmNoX0dvIiwibmJmIjoxNzIzNjQ5ODQzLCJ1c2VyX2lkIjoiNCJ9.hHnq9rEU_NhlJVNojhi7aZgD48vH6L_nBhZIMPhH_giv4odU6xtoPcwA0NEDp55Bc7JJhNRi4Z-3MRJj4TjFbWTJQPK8NQhvW1V4G2wEn2LDvokN0Plw65NPtKD4ARSnIvD6onOjH8ryjU1pg2X2IVZKc8pwvmHpZvavsJC4eY2_1sUVoj04fyCVBnzLg5C6ddNdXL_87oX8-mbZWqhoZlaD5-IvxBWlFBCsX3LagkceNWNxOihGi1t4KQJyFNYJcoVxb2i9bFLzbggUk2bLvJkzAj8LcnBm8g4hTnx9KlI7Q6RLYAcwwmY9qAGb4t3NCEzMeg_URpSw5OD9fMF8tMBRXvESs4UhDBCMeQfsarFC9oLM7-h2ohOF5eEglB13pmj3PFK2rHxLv2y1cyqu9UaMVh-TRCDJMzfGTj5TX_mCCHusJecZQl6aEh1C_Ta85xFeht1KjNnECKWg30C9HjxE0j6DP9t1dNvNgfXdBOjSYmugk7_yr2iPnMSOg-lwYz18dZC709l3M1QEmFmW7UZkGjwVs_WLozZ1tmvg-X_e0f8bNJUkMkToDu9DIm9pZBoST-Fz3T7y_SXyU-Yy3yYntVaOjzpTA2wG18jlfvNtqsbEv1lbWBTulxedz2HiDE4vW_Gq-RmK2prUdL8gdaen7cRES0lrMpEpwnFwjgg",
            "Content-Type": "application/json"
        }
        data = {
            "title": title,
            "content": content,
            "userId": user_id,
            "docId": doc_id
        }
        response = requests.post("http://localhost:5173/user-notis", headers=headers, json=data)
        response.raise_for_status()
        return JSONResponse(status_code=200, content={"status": "200", "message": "Notification sent successfully"})
    except Exception as e:
        logging.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail="Error sending notification")


@router.put("/edit")
async def edit_document(request: Request, doc_id: int,
                        description: str = Body(None),
                        category_id: int = Body(None)):
    document = document_service.get_document_by_id(doc_id)
    user_id = get_request_user_id(request)
    user = document_service.get_user_by_id(user_id)
    if not user.is_organization_manager and not user.is_dept_manager:
        return JSONResponse(status_code=403,
                            content={"status": "403", "message": "You do not have permission to insert document"})
    category = document_service.get_category_by_id(document.category_id)
    if category.organization_id != user.organization_id:
        return JSONResponse(status_code=403, content={"status": "403",
                                                      "message": "You do not have permission to insert document into this category"})
    else:
        if not user.is_organization_manager:
            if category.department_id != user.dept_id:
                return JSONResponse(status_code=403, content={"status": "403",
                                                              "message": "You do not have permission to insert document into this category"})
    if description is not None or category_id is not None:
        with session_scope() as session:
            if description is not None:
                document.description = description
            if category_id is not None:
                document.category_id = category_id
                update_category_in_elasticsearch(document.id, category_id, user_id)
            session.merge(document)
            session.commit()

    serialized_document = serialize_document(document)
    return JSONResponse(status_code=200,
                        content={"status": "200", "message": "Document updated successfully",
                                 "data": serialized_document})


@router.delete("/delete")
def delete_document(request: Request, doc_id: int):
    user_id = get_request_user_id(request)
    try:
        document = document_service.delete_document(doc_id, user_id)
        document_dict = document_service.serialize_document(document)  # Convert Document object to dict
        return JSONResponse(status_code=200, content={"status": "200", "message": "Document deleted successfully",
                                                      "data": document_dict})
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"status": str(e.status_code), "message": e.detail})


@router.post("/move_to_trash")
def move_to_trash(request: Request, doc_id: int):
    user_id = get_request_user_id(request)
    document = document_service.get_document_by_id(doc_id)
    user = document_service.get_user_by_id(user_id)
    if not user.is_organization_manager and not user.is_dept_manager:
        return JSONResponse(status_code=403,
                            content={"status": "403", "message": "You do not have permission to insert document"})
    category = document_service.get_category_by_id(document.category_id)
    if category.organization_id != user.organization_id:
        return JSONResponse(status_code=403, content={"status": "403",
                                                      "message": "You do not have permission to insert document into this category"})
    else:
        if not user.is_organization_manager:
            if category.department_id != user.dept_id:
                return JSONResponse(status_code=403, content={"status": "403",
                                                              "message": "You do not have permission to insert document into this category"})
    document_dict = document_service.move_to_trash(doc_id)
    return JSONResponse(status_code=200, content={"status": "200", "message": "Document moved to trash successfully",
                                                  "data": document_dict})


@router.get("/view")
async def view_document(request: Request, doc_id: int):
    user_id = get_request_user_id(request)
    document = document_service.view_document(doc_id, user_id)
    return JSONResponse(status_code=200, content={"status": "200", "message": "Document retrieved successfully",
                                                  "data": document})
    # if not document.file_id:
    #     return JSONResponse(content={"error": "File not found"}, status_code=404)

    # try:
    #     request = GoogleRequest()
    #     credentials.refresh(request)
    #
    #     file_metadata = drive_service.files().get(fileId=document.file_id, fields='mimeType').execute()
    #     mime_type = file_metadata.get('mimeType')
    #
    #     if mime_type in ['application/vnd.google-apps.document', 'application/vnd.google-apps.spreadsheet',
    #                      'application/vnd.google-apps.presentation']:
    #         if mime_type == 'application/vnd.google-apps.document':
    #             export_mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'  # DOCX
    #         elif mime_type == 'application/vnd.google-apps.spreadsheet':
    #             export_mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'  # XLSX
    #         elif mime_type == 'application/vnd.google-apps.presentation':
    #             export_mime_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'  # PPTX
    #         request = drive_service.files().export_media(fileId=document.file_id, mimeType=export_mime_type)
    #     else:
    #         request = drive_service.files().get_media(fileId=document.file_id)
    #
    #     file_content = io.BytesIO(request.execute())
    #
    #     if mime_type.startswith('image/') or mime_type.startswith('video/') or mime_type == 'text/plain':
    #         encoded_content = base64.b64encode(file_content.getvalue()).decode('utf-8')
    #     else:
    #         try:
    #             encoded_content = file_content.getvalue().decode('utf-8')
    #         except UnicodeDecodeError:
    #             encoded_content = base64.b64encode(file_content.getvalue()).decode('utf-8')
    #
    #     return JSONResponse(content={"mimeType": mime_type, "content": encoded_content})
    #
    # except Exception as e:
    #     error_info = traceback.format_exc()
    #     logging.error(f"Unhandled error: {error_info}")
    #     return JSONResponse(content={"error": str(e)}, status_code=500)


# @router.get("/viewfile/{file_id}")
# async def view_document(file_id: str):
#     if not file_id:
#         raise HTTPException(status_code=404, detail="Không tìm thấy tệp")
#
#     try:
#         request = GoogleRequest()
#         credentials.refresh(request)
#
#         file_metadata = drive_service.files().get(fileId=file_id, fields='mimeType').execute()
#         mime_type = file_metadata.get('mimeType')
#
#         if mime_type in [
#             'application/vnd.google-apps.document',
#             'application/vnd.google-apps.spreadsheet',
#             'application/vnd.google-apps.presentation'
#         ]:
#             export_mime_type = {
#                 'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
#                 # DOCX
#                 'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
#                 # XLSX
#                 'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
#                 # PPTX
#             }[mime_type]
#             request = drive_service.files().export_media(fileId=file_id, mimeType=export_mime_type)
#         else:
#             request = drive_service.files().get_media(fileId=file_id)
#
#         file_content = io.BytesIO(request.execute())
#         return StreamingResponse(file_content, media_type=mime_type)
#
#     except Exception as e:
#         error_info = traceback.format_exc()
#         logging.error(f"Lỗi không xử lý được: {error_info}")
#         raise HTTPException(status_code=500, detail="Lỗi máy chủ nội bộ")


@router.get("/get_by_category")
async def get_document_by_category(request: Request, category_id: int, page: int = 1, page_size: int = 10):
    user_id = get_request_user_id(request)
    documents, page, total_pages = await document_service.get_documents_by_category_service(category_id, page,
                                                                                            page_size,
                                                                                            user_id)
    return JSONResponse(status_code=200, content={
        "status": "200",
        "message": "Documents retrieved successfully",
        "data": documents,
        "page": page,
        "total_pages": total_pages
    })


@router.get("/get_recent_files")
async def get_recent_files(request: Request, time: date, page: int = 1, page_size: int = 30):
    user_id = get_request_user_id(request)
    documents, page, total_pages = document_service.get_recent_documents_service(time, user_id, page, page_size)
    serialized_documents = [document_service.serialize_document(doc) for doc in documents]
    return JSONResponse(status_code=200, content={
        "status": "200",
        "message": "Documents retrieved successfully",
        "data": serialized_documents,
        "page": page,
        "total_pages": total_pages
    })


@router.get("/get_recent_files_deleted")
def get_recent_documents_deleted(request: Request, time: date, page: int = 1, page_size: int = 10):
    user_id = get_request_user_id(request)
    documents, page, total_pages = document_service.get_recent_documents_deleted_service(time, user_id, page, page_size)
    return JSONResponse(status_code=200, content={
        "status": "200",
        "message": "Documents retrieved successfully",
        "data": documents,
        "page": page,
        "total_pages": total_pages
    })


@router.get("/recover_files")
def recover_files(request: Request, doc_id: int):
    user_id = get_request_user_id(request)
    user = document_service.get_user_by_id(user_id)
    document = document_service.get_document_by_id(doc_id)
    if not user.is_organization_manager and not user.is_dept_manager:
        return JSONResponse(status_code=403,
                            content={"status": "403", "message": "You do not have permission to insert document"})
    category = document_service.get_category_by_id(document.category_id)
    if category.organization_id != user.organization_id:
        return JSONResponse(status_code=403, content={"status": "403",
                                                      "message": "You do not have permission to insert document into this category"})
    else:
        if not user.is_organization_manager:
            if category.department_id != user.dept_id:
                return JSONResponse(status_code=403, content={"status": "403",
                                                              "message": "You do not have permission to insert document into this category"})
    document = document_service.recover_files(doc_id, user_id)
    return JSONResponse(status_code=200,
                        content={"status": "200", "message": "Document recovered successfully", "data": document})


class RequestSearchBody(BaseModel):
    query: str


@router.get("/search")
async def find_doc_nlp(request: Request,
                       category_id: Union[str] = None,
                       type_doc: Union[str] = None,
                       question: Union[str] = "", page_num: int = 1,
                       page_size: int = 10):
    user_id = get_request_user_id(request)
    user_id = get_request_user_id(request)
    start = (page_num - 1) * page_size
    end = page_size

    docs, total, total_pages, normalized_question = document_service.search_documents(question, client, category_id,
                                                                                      type_doc, start, end,
                                                                                      user_id)
    docs = [doc for doc in docs if doc['relevance_score'] is not None]
    docs = sorted(docs, key=lambda d: d['relevance_score'], reverse=True)

    return JSONResponse(status_code=200, content={
        "status": 200,
        "data": docs,
        "total": total,
        "normalized_question": normalized_question,
        "currentPage": page_num,
        "totalPage": total_pages
    })


@router.get("/suggest_metadata")
async def suggest_metadata(request: Request, document_number: str = None, issuing_authority: str = None,
                           date_of_issuance: str = None, signature: str = None, agency_address: str = None):
    user_id = get_request_user_id(request)

    metadata = document_service.filter_metadata_by_user(user_id, document_number=document_number,
                                                        issuing_authority=issuing_authority,
                                                        date_of_issuance=date_of_issuance,
                                                        signature=signature,
                                                        agency_address=agency_address)

    if not metadata:
        return JSONResponse(status_code=200, content={"status": 200, "data": []})

    field_data = {
        "document_number": set(),
        "issuing_authority": set(),
        "date_of_issuance": set(),
        "signature": set(),
        "agency_address": set()
    }

    for item in metadata:
        for field in field_data.keys():
            if field in item:
                field_data[field].add(item[field])

    response_data = {}
    if document_number is not None:
        response_data["document_number"] = list(field_data["document_number"])
    if issuing_authority is not None:
        response_data["issuing_authority"] = list(field_data["issuing_authority"])
    if date_of_issuance is not None:
        response_data["date_of_issuance"] = list(field_data["date_of_issuance"])
    if signature is not None:
        response_data["signature"] = list(field_data["signature"])
    if agency_address is not None:
        response_data["agency_address"] = list(field_data["agency_address"])

    return JSONResponse(status_code=200, content={"status": 200, "data": response_data})


@router.get("/get_document_by_metadata")
async def get_document_by_metadata(request: Request, document_number: str = None, issuing_authority: str = None,
                                   date_of_issuance: str = None, signature: str = None, agency_address: str = None):
    user_id = get_request_user_id(request)
    if all(param is None for param in
           [document_number, issuing_authority, date_of_issuance, signature, agency_address]):
        return JSONResponse(status_code=200, content={
            "status": 200,
            "data": []
        })

    metadata = document_service.get_document_by_meta(user_id, document_number=document_number,
                                                     issuing_authority=issuing_authority,
                                                     date_of_issuance=date_of_issuance, signature=signature,
                                                     agency_address=agency_address)
    return JSONResponse(status_code=200, content={
        "status": 200,
        "data": metadata
    })


@router.get("/set_password_document")
async def set_password_document(request: Request, doc_id: int, password: str):
    user_id = get_request_user_id(request)
    user = document_service.get_user_by_id(user_id)
    if not user.is_organization_manager and not user.is_dept_manager:
        return JSONResponse(status_code=403,
                            content={"status": "403", "message": "You do not have permission to insert document"})
    category = document_service.get_category_by_doc_id(doc_id)
    if category.organization_id != user.organization_id:
        return JSONResponse(status_code=403, content={"status": "403",
                                                      "message": "You do not have permission to insert document into this category"})
    else:
        if not user.is_organization_manager:
            if category.department_id != user.dept_id:
                return JSONResponse(status_code=403, content={"status": "403",
                                                              "message": "You do not have permission to insert document into this category"})
    document_service.set_password_document(doc_id, password)
    return JSONResponse(status_code=200,
                        content={"status": "200", "message": "Document password set successfully"})


@router.get("/remove_password_document")
async def remove_password_document(request: Request, doc_id: int):
    user_id = get_request_user_id(request)
    user = document_service.get_user_by_id(user_id)
    if not user.is_organization_manager and not user.is_dept_manager:
        return JSONResponse(status_code=403,
                            content={"status": "403", "message": "You do not have permission to insert document"})
    category = document_service.get_category_by_doc_id(doc_id)
    if category.organization_id != user.organization_id:
        return JSONResponse(status_code=403, content={"status": "403",
                                                      "message": "You do not have permission to insert document into this category"})
    else:
        if not user.is_organization_manager:
            if category.department_id != user.dept_id:
                return JSONResponse(status_code=403, content={"status": "403",
                                                              "message": "You do not have permission to insert document into this category"})
    document = document_service.remove_password_document(doc_id)
    return JSONResponse(status_code=200,
                        content={"status": "200", "message": "Document password removed successfully"})


@router.get("/check_document_password")
async def check_document_password(doc_id: int, password: str):
    document = document_service.check_document_password(doc_id, password)
    return JSONResponse(status_code=200,
                        content={"status": "200", "message": "Document password checked successfully",
                                 "data": document})


@router.get("/check_valid_document_password")
async def check_valid_document_password(doc_id: int):
    document = document_service.check_valid_document_password(doc_id)
    return JSONResponse(status_code=200,
                        content={"status": "200", "message": "Document password checked successfully",
                                 "data": document})


@router.get("/get_document_by_orgnization")
async def get_document_by_orgnization(request: Request, organization_id: int):  # noqa
    document_ids = document_service.get_documents_by_organization_service(organization_id)
    return JSONResponse(status_code=200, content={
        "status": 200,
        "data": document_ids,
    })


@router.post("/photos/insert")
async def generate_image_description_endpoint(doc_id: int, file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        description = generate_image_description(doc_id, file_content)
        return {"description": description}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get("/search_photo")
async def search_image_descriptions_endpoint(request: Request, query: str):
    user_id = get_request_user_id(request)
    try:
        documents = search_image_descriptions(query, user_id)
        serialized_documents = [serialize_document(doc) for doc in documents]
        return {"documents": serialized_documents}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get("/check_plagiarism")
async def check_plagiarism_endpoint(request: Request, doc_id: int):
    user_id = get_request_user_id(request)
    try:
        result = check_plagiarism(doc_id)
        return JSONResponse(status_code=200, content={"status": "200", "data": result})
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.get("/compare_documents")
async def compare_documents(request: Request, doc_id1: int, doc_id2: int):
    user_id = get_request_user_id(request)
    try:
        result = await async_compare_document_similarity(doc_id1, doc_id2)
        return JSONResponse(status_code=200, content={"status": "200", "data": result})
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


async def async_compare_document_similarity(doc_id1: int, doc_id2: int):
    return compare_document_similarity(doc_id1, doc_id2)


@router.post("/compare-images")
def compare_images(request: Request, doc_id1: int, doc_id2: int):
    user_id = get_request_user_id(request)
    image_base64_1 = get_image_from_db(doc_id1)
    image_base64_2 = get_image_from_db(doc_id2)

    image1 = create_image_from_base64(image_base64_1)
    image2 = create_image_from_base64(image_base64_2)

    if image1 is None or image2 is None:
        raise HTTPException(status_code=400, detail="One or both images could not be decoded")

    # Ensure both images have the same dimensions
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    ssim_score = calculate_ssim(image1, image2)

    return {
        "similarity": ssim_score,
    }


@router.put("/edit_metadata")
async def edit_metadata(request: Request, document_id: int, document_number: str = None, issuing_authority: str = None,
                        date_of_issuance: str = None, signature: str = None, agency_address: str = None):
    user_id = get_request_user_id(request)
    try:
        updated_metadata = document_service.update_metadata(document_id, document_number=document_number,
                                                            issuing_authority=issuing_authority,
                                                            date_of_issuance=date_of_issuance, signature=signature,
                                                            agency_address=agency_address)
        return JSONResponse(status_code=200, content={
            "status": 200,
            "data": updated_metadata
        })
    except Exception as e:
        logging.error(f"Error updating metadata: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@router.put("/edit_description_photo")
async def edit_description_photo(request: Request, doc_id: int, new_description: str):
    user_id = get_request_user_id(request)
    try:
        updated_description = document_service.update_photo_description(doc_id, new_description)
        return JSONResponse(status_code=200, content={
            "status": 200,
            "data": updated_description
        })
    except Exception as e:
        logging.error(f"Error updating photo description: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")