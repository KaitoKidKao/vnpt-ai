"""Script to generate expanded test data and knowledge base for comprehensive pipeline testing."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

def generate_knowledge_base() -> None:
    """Generate expanded knowledge_base.txt for RAG ingestion."""
    knowledge = """# Lịch sử Việt Nam

## Cách mạng tháng Tám 1945
Cách mạng tháng Tám là cuộc cách mạng giành chính quyền do Đảng Cộng sản Đông Dương lãnh đạo, diễn ra vào tháng 8 năm 1945. Ngày 2 tháng 9 năm 1945, Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Quảng trường Ba Đình, khai sinh nước Việt Nam Dân chủ Cộng hòa.

## Chiến thắng Điện Biên Phủ 1954
Chiến thắng Điện Biên Phủ diễn ra vào năm 1954, đánh dấu sự kết thúc của cuộc kháng chiến chống thực dân Pháp. Chiến dịch kéo dài 56 ngày đêm, từ ngày 13 tháng 3 đến ngày 7 tháng 5 năm 1954.

## Khởi nghĩa Yên Bái 1930
Khởi nghĩa Yên Bái do Việt Nam Quốc dân Đảng tổ chức, nổ ra vào đêm 9 rạng sáng ngày 10 tháng 2 năm 1930 tại Yên Bái và một số tỉnh Bắc Kỳ.

# Địa lý Việt Nam

## Thủ đô Hà Nội
Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam. Thành phố nằm ở vùng đồng bằng sông Hồng, có lịch sử hơn 1000 năm văn hiến. Đặc sản nổi tiếng nhất là Phở, Bún chả, Cốm làng Vòng.

## Thành phố Hồ Chí Minh
Thành phố Hồ Chí Minh (trước đây là Sài Gòn) là thành phố lớn nhất Việt Nam về dân số và kinh tế, nằm ở miền Nam Việt Nam.

## Đà Nẵng
Đà Nẵng là thành phố trực thuộc trung ương, nằm ở miền Trung Việt Nam, được mệnh danh là thành phố đáng sống nhất Việt Nam.

# Văn hóa Việt Nam

## Áo dài
Áo dài là trang phục truyền thống của Việt Nam, được coi là quốc phục. Áo dài thường được mặc trong các dịp lễ hội, cưới hỏi và các sự kiện quan trọng.

## Tết Nguyên đán
Tết Nguyên đán là lễ hội lớn nhất trong năm của người Việt Nam, diễn ra vào đầu năm âm lịch. Đây là dịp để gia đình đoàn tụ và thờ cúng tổ tiên.

# Pháp luật & An ninh mạng

## Luật An ninh mạng 2018
Luật An ninh mạng năm 2018 quy định về hoạt động bảo vệ an ninh quốc gia và bảo đảm trật tự, an toàn xã hội trên không gian mạng.
Một trong những quy định quan trọng là yêu cầu các doanh nghiệp cung cấp dịch vụ trên mạng viễn thông, mạng internet tại Việt Nam phải **lưu trữ dữ liệu người sử dụng tại Việt Nam** (Data Localization).
Các hành vi bị nghiêm cấm bao gồm: Sử dụng không gian mạng để tuyên truyền chống Nhà nước; Kích động bạo loạn; Đăng tải thông tin sai sự thật gây hoang mang dư luận; Xúc phạm danh dự, nhân phẩm người khác.

# Khoa học & Công nghệ

## Giải thưởng VinFuture
VinFuture là giải thưởng khoa học công nghệ toàn cầu do Việt Nam khởi xướng. 
Mùa giải đầu tiên (2021), Giải thưởng Chính trị giá 3 triệu USD đã được trao cho các nhà khoa học: Katalin Karikó, Drew Weissman và Pieter Cullis với công trình nghiên cứu về **công nghệ mRNA**, mở đường cho việc sản xuất thành công vắc-xin Covid-19 hiệu quả.

## Trí tuệ nhân tạo (AI) tại Việt Nam
Việt Nam đang đẩy mạnh nghiên cứu và ứng dụng AI trong nhiều lĩnh vực như y tế, giao thông, và giáo dục. Chính phủ đã ban hành Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng Trí tuệ nhân tạo đến năm 2030.
"""
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / "knowledge_base.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(knowledge)

    print(f"[DataGen] Generated: {output_path}")


if __name__ == "__main__":
    generate_knowledge_base()
    print("[DataGen] Dummy data generation completed!")