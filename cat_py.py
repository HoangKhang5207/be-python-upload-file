import os

output_file = 'D:/NgoHoangHuy/HopTacPhanMem/WebDoc/Be-python/xuly.txt'
base_dir = 'D:/NgoHoangHuy/HopTacPhanMem/WebDoc/Be-python'

# Biến đếm số file .py
file_count = 0

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_count += 1  # Tăng biến đếm
                filepath = os.path.join(root, file)
                outfile.write(f"--- File: {filepath} ---\n")
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                    outfile.write("\n\n")  # Thêm khoảng cách giữa các file
                except Exception as e:
                    outfile.write(f"Error reading file {filepath}: {e}\n\n")
    
    # Ghi tổng số file .py vào cuối file đầu ra
    outfile.write(f"\n--- Tổng số file .py được nối: {file_count} ---\n")

print(f"Đã nối {file_count} file .py từ {base_dir} và các thư mục con vào {output_file}")