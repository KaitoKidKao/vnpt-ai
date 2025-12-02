"""Script to generate dummy test data and knowledge base for development."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

def generate_public_test_csv() -> None:
    """Generate dummy public_test.csv with logic, math, and reasoning questions."""
    questions = [
        # --- NHÓM 1: CÂU HỎI KIẾN THỨC CƠ BẢN ---
        {
            "id": "Q001",
            "question": "Năm 1945, sự kiện lịch sử quan trọng nào đã diễn ra ở Việt Nam?",
            "A": "Khởi nghĩa Yên Bái",
            "B": "Cách mạng tháng Tám thành công",
            "C": "Hiệp định Genève được ký kết",
            "D": "Chiến thắng Điện Biên Phủ",
            "category": "history",
        },
        {
            "id": "Q002",
            "question": "Thủ đô của Việt Nam là thành phố nào?",
            "A": "Hồ Chí Minh",
            "B": "Đà Nẵng",
            "C": "Hà Nội",
            "D": "Huế",
            "category": "geography",
        },
        # --- NHÓM 2: TOÁN HỌC & GIẢI PHƯƠNG TRÌNH ---
        {
            "id": "Q003",
            "question": "Một nông trại có cả gà và chó. Tổng số đầu là 36, tổng số chân là 100. Hỏi có bao nhiêu con chó?",
            "A": "12",
            "B": "14",
            "C": "22",
            "D": "16",
            "category": "math_logic",
        },
        {
            "id": "Q004",
            "question": "Tìm nghiệm dương của phương trình: x^2 - 5x + 6 = 0",
            "A": "2 và 3",
            "B": "1 và 6",
            "C": "-2 và -3",
            "D": "Chỉ có 3",
            "category": "math_equation",
        },
        # --- NHÓM 3: TÍNH TOÁN SỐ HỌC LỚN ---
        {
            "id": "Q005",
            "question": "Giá trị của biểu thức S = 1 + 2 + 3 + ... + 100 là bao nhiêu?",
            "A": "5000",
            "B": "5050",
            "C": "5100",
            "D": "4950",
            "category": "math_sequence",
        },
        {
            "id": "Q006",
            "question": "Tính kết quả chính xác của phép nhân: 123456 x 789",
            "A": "97406784",
            "B": "97406794",
            "C": "97306784",
            "D": "97506784",
            "category": "math_arithmetic",
        },
        {
            "id": "Q007",
            "question": "Nếu gửi tiết kiệm 100 triệu đồng với lãi suất 8%/năm (lãi kép, nhập gốc hàng năm), sau 10 năm sẽ nhận được bao nhiêu tiền (làm tròn đến triệu đồng gần nhất)?",
            "A": "216 triệu",
            "B": "180 triệu",
            "C": "200 triệu",
            "D": "250 triệu",
            "category": "math_finance",
        },
        # --- NHÓM 4: LOGIC QUY LUẬT & TỔ HỢP ---
        {
            "id": "Q008",
            "question": "Tìm số tiếp theo trong dãy số: 2, 6, 12, 20, 30, ...?",
            "A": "38",
            "B": "40",
            "C": "42",
            "D": "44",
            "category": "logic_sequence",
        },
        {
            "id": "Q009",
            "question": "Trong một phòng họp có 10 người. Nếu mỗi người đều bắt tay với tất cả những người khác đúng một lần, thì có tổng cộng bao nhiêu cái bắt tay?",
            "A": "90",
            "B": "100",
            "C": "45",
            "D": "50",
            "category": "math_combinatorics",
        },
        {
            "id": "Q010",
            "question": "Nếu hôm nay là Thứ Hai, thì 1000 ngày sau (tính từ ngày mai) là thứ mấy trong tuần?",
            "A": "Thứ Tư",
            "B": "Thứ Năm",
            "C": "Chủ Nhật",
            "D": "Thứ Ba",
            "category": "logic_datetime",
        },
    ]

    # Đảm bảo thư mục tồn tại
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = DATA_DIR / "public_test.csv"
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "question", "A", "B", "C", "D", "category"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions)

    print(f"Generated: {output_path} with {len(questions)} questions.")


def generate_knowledge_base() -> None:
    """Generate dummy knowledge_base.txt for RAG ingestion."""
    knowledge = """# Lịch sử Việt Nam

## Cách mạng tháng Tám 1945
Cách mạng tháng Tám là cuộc cách mạng giành chính quyền do Đảng Cộng sản Đông Dương lãnh đạo, diễn ra vào tháng 8 năm 1945. Ngày 2 tháng 9 năm 1945, Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Quảng trường Ba Đình, khai sinh nước Việt Nam Dân chủ Cộng hòa.

## Chiến thắng Điện Biên Phủ 1954
Chiến thắng Điện Biên Phủ diễn ra vào năm 1954, đánh dấu sự kết thúc của cuộc kháng chiến chống thực dân Pháp. Chiến dịch kéo dài 56 ngày đêm, từ ngày 13 tháng 3 đến ngày 7 tháng 5 năm 1954.

## Khởi nghĩa Yên Bái 1930
Khởi nghĩa Yên Bái do Việt Nam Quốc dân Đảng tổ chức, nổ ra vào đêm 9 rạng sáng ngày 10 tháng 2 năm 1930 tại Yên Bái và một số tỉnh Bắc Kỳ.

# Địa lý Việt Nam

## Thủ đô Hà Nội
Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam. Thành phố nằm ở vùng đồng bằng sông Hồng, có lịch sử hơn 1000 năm văn hiến.

## Thành phố Hồ Chí Minh
Thành phố Hồ Chí Minh (trước đây là Sài Gòn) là thành phố lớn nhất Việt Nam về dân số và kinh tế, nằm ở miền Nam Việt Nam.

## Đà Nẵng
Đà Nẵng là thành phố trực thuộc trung ương, nằm ở miền Trung Việt Nam, được mệnh danh là thành phố đáng sống nhất Việt Nam.

# Văn hóa Việt Nam

## Áo dài
Áo dài là trang phục truyền thống của Việt Nam, được coi là quốc phục. Áo dài thường được mặc trong các dịp lễ hội, cưới hỏi và các sự kiện quan trọng.

## Tết Nguyên đán
Tết Nguyên đán là lễ hội lớn nhất trong năm của người Việt Nam, diễn ra vào đầu năm âm lịch. Đây là dịp để gia đình đoàn tụ và thờ cúng tổ tiên.
"""
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = DATA_DIR / "knowledge_base.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(knowledge)

    print(f"Generated: {output_path}")


if __name__ == "__main__":
    generate_public_test_csv()
    generate_knowledge_base()
    print("Dummy data generation completed!")