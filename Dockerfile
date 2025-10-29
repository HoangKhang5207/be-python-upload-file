# Sử dụng image cơ sở nhỏ hơn
FROM python:3.8-slim-bullseye

# Cài đặt gcc, poppler-utils, tesseract-ocr, libgl1-mesa-glx, libglib2.0-0 và các thư viện liên quan
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Đặt biến môi trường TESSDATA_PREFIX
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt và cài đặt các gói cần thiết
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Tạo user và group mới
RUN groupadd -g 1002 app_group && \
    useradd -g app_group --uid 1002 app_user

# Tạo thư mục home cho app_user và đặt quyền sở hữu
RUN mkdir -p /home/app_user && chown -R app_user:app_group /home/app_user

# Tạo thư mục cho file và đặt quyền sở hữu
RUN mkdir -p /app/file && chown -R app_user:app_group /app/file

# Sao chép file JSON chứa thông tin xác thực cho Google Drive và đặt quyền sở hữu
COPY deft-chariot-432114-g3-c3e424e1da09.json .
RUN chown app_user:app_group deft-chariot-432114-g3-c3e424e1da09.json

# Sao chép thư mục models và đặt quyền sở hữu
COPY models /app/models
RUN chown -R app_user:app_group /app/models

# Sao chép mã nguồn và thay đổi quyền sở hữu
COPY . .
RUN chown -R app_user:app_group /app

# Chuyển sang user mới
USER app_user

# Thiết lập lệnh mặc định khi chạy container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]