# master-thesis-mooc-recommendation
# Mô hình Khuyến nghị Khóa học MOOC có thể Giải thích (Explainable MOOC Recommendation)

Đồ án Thạc sĩ Khoa học máy tính (Mã số: 8480101) thực hiện tại Trường Đại học Tư thục Quốc tế Sài Gòn (SIU).

* **Học viên:** Vũ Nhân Khánh  
* **Người hướng dẫn khoa học:** TS. Huỳnh Ngọc Tín & TS. Hứa Cẩm Hào  

---

## 📌 Giới thiệu tổng quan
Kho lưu trữ này chứa toàn bộ mã nguồn triển khai thực nghiệm hệ thống **Khuyến nghị khóa học MOOC có thể giải thích** dựa trên tổ hợp **Đồ thị tri thức (Knowledge Graph - KG)** và **Học tăng cường (Reinforcement Learning - RL)**, kết hợp cùng Mô hình ngôn ngữ lớn **Qwen2.5-7B-Instruct** để biên dịch lời giải thích tự nhiên.

Mô hình kế thừa nguyên lý của kiến trúc PGPR và UPGPR, đồng thời mang đến các cải tiến:
* **Cơ chế Hàm phần thưởng đa thành phần cải tiến:** $R_{step} = R_{target} + R_{semantic} + R_{pedagogical}$ biến thiên liên tục trong khoảng `[0.0, 11.5]`.
* **Mạng chính sách MLP 2 tầng** với hàm kích hoạt ELU tối ưu hóa số lượng tham số, đảm bảo tác nhân hội tụ nhanh chóng.
* **Cơ chế Đảm bảo tính xác thực cấu trúc (Structural Grounding)** ép mô hình ngôn ngữ lớn LLM sinh lời giải thích tự nhiên dựa trên các thực thể chứng thực có thật từ đồ thị tri thức.

---

## 📁 Cấu trúc mã nguồn (Pipeline Thực nghiệm)

Mã nguồn được thiết kế theo dạng đường ống (Pipeline) tuần tự, chia nhỏ thành 6 tệp Notebook chạy trên môi trường Google Colab (GPU Nvidia T4):

1. **`01_Data_Preprocessing.ipynb`**
   * Đọc và tiền xử lý bộ dữ liệu giáo dục `MOOCCube`.
   * Áp dụng kỹ thuật lọc K-core với $k=10$ loại bỏ các tương tác thưa thớt.
   * Phân chia dữ liệu tương tác đăng ký (Enrollment) theo định danh người học (User-based Split): 80% Train, 10% Validation, 10% Test.

2. **`02_KG_Construction.ipynb`**
   * Xây dựng cấu trúc Đồ thị tri thức Giáo dục dị thể (HIN).
   * Thiết lập 4 loại thực thể (`User`, `Course`, `Concept`, `Teacher`) và 8 loại quan hệ hai chiều hoàn chỉnh (`enrolled`, `teaches`, `has_concept`, `belongs_to`, `provides` và các quan hệ ngược).

3. **`03_Embedding_Training.ipynb`**
   * Huấn luyện không gian nhúng hình học cho các thực thể và quan hệ trên đồ thị bằng mô hình dịch chuyển **TransE** với $d=100$.

4. **`04_RL_Training_Eval.ipynb`**
   * Xây dựng môi trường Quá trình quyết định Markov (MDP).
   * Khởi tạo tác nhân RL Agent và huấn luyện mạng chính sách MLP bằng thuật toán Policy Gradient (REINFORCE cải tiến) phối hợp với Cơ chế Hàm phần thưởng đa thành phần.

5. **`05_RL_Evaluation.ipynb`**
   * Chạy thuật toán Tìm kiếm chùm (Beam Search) khống chế số bước đi tối đa (`MAX_STEPS = 4`) để trích xuất danh sách Top-10 khóa học khuyến nghị.
   * Tính toán các chỉ số đo lường định lượng: Hit Rate (HR), NDCG, Precision, và Recall.
   * Kết xuất cấu trúc vết đường đi (Symbolic Paths) dẫn đến khóa học đích thành công.

6. **`06_LLM_Generation.ipynb`**
   * Tích hợp mô hình ngôn ngữ lớn **Qwen2.5-7B-Instruct** (kỹ thuật lượng hóa 4-bit NF4) thông qua kỹ nghệ gợi ý nhận biết ngữ cảnh (Context-aware Prompt Engineering).
   * Biên dịch các đường đi ký hiệu thô thành câu giải thích tự nhiên bằng tiếng Việt có ràng buộc thực thể cấu trúc.

---

## 📊 Bộ dữ liệu (Dataset)
Mô hình thực nghiệm trên bộ dữ liệu giáo dục quy mô lớn **MOOCCube**. Vì lý do dung lượng và bản quyền dữ liệu, tệp dữ liệu thô không được đính kèm trực tiếp trong kho lưu trữ này. 
* Người dùng cần tải bộ dữ liệu gốc từ nguồn công khai tại: [Tsinghua MOOCCube](https://github.com/thunlp/MOOCCube).
* Sau khi tải về, vui lòng giải nén và đặt các tệp tin vào thư mục `data/` trước khi tiến hành chạy tệp `01_Data_Preprocessing.ipynb`.

---

## 🛠️ Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường
Kho mã nguồn yêu cầu cài đặt Python $\ge 3.8$ và các thư viện học máy cốt lõi. Hãy tạo môi trường ảo và cài đặt qua tệp `requirements.txt`:

pip install -r requirements.txt

(Các thư viện chính bao gồm: torch, torch-geometric, transformers, bitsandbytes, accelerate, numpy, pandas, scikit-learn).

### 2. Tiến trình thực thi
Để tái lập lại các kết quả nghiên cứu trong đồ án, vui lòng tải bộ dữ liệu MOOCCube về, giải nén và đặt các tệp tin vào thư mục `data/` trước khi tiến hành chạy toàn bộ cấu trúc các tệp từ 01_... đến 06_... trên môi trường Google Colab theo đúng thứ tự đánh số.

Lưu ý: Đối với bước sinh lời giải thích tự nhiên tại tệp số 06, hãy đảm bảo hệ thống có kết nối mạng ổn định để tải cấu hình trọng số mô hình Qwen từ Hugging Face.

```bash
