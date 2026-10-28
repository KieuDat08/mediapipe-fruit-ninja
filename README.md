# 🍉 Hand-Tracking Fruit Ninja

Trò chơi **chém hoa quả bằng ngón tay** trước webcam, xây dựng với **OpenCV + MediaPipe Hands**. Bài tập lớn môn Lập trình Python: *Xây dựng ứng dụng tương tác với Camera sử dụng MediaPipe*.

```
Camera -> MediaPipe Hands -> Landmark 8 -> Thuật toán -> Game
```

## 🎬 Video Demo

[Link demo Mediapipe](https://drive.google.com/drive/folders/1M7xQbj5EhKY1qLMgh7bGXuElfGWO_sx-?usp=sharing)
## ✨ Tính năng

- Điều khiển hoàn toàn bằng **ngón trỏ**, không cần chuột/bàn phím.
- **Vệt kiếm** phát sáng dựa trên `deque` lưu các vị trí gần nhất.
- Chỉ tính là chém khi **vung đủ nhanh** (ngưỡng vận tốc), tay đứng yên không ăn điểm.
- Hoa quả bay theo **quỹ đạo parabol** (vật lý ném xiên).
- Quả **tách làm hai nửa** và bắn nước khi bị chém.
- Đếm số quả đã chém, hệ thống **3 mạng**, màn hình **Game Over**, nhấn `R` để chơi lại.
- Độ khó tăng dần theo điểm số.
- Không cần ảnh/âm thanh ngoài: toàn bộ đồ họa vẽ bằng hàm hình học của OpenCV.

## 🧠 Cấu trúc thuật toán tóm tắt

| Bước | Nội dung |
|---|---|
| 1. Chuyển tọa độ | `(x, y)` chuẩn hóa [0,1] -> pixel: `px = x·W`, `py = y·H` |
| 2. Vận tốc chém | `v = khoảng cách Euclid giữa 2 frame / Δt`, làm mượt EMA, so với ngưỡng |
| 3. Vật lý quả | `vy += g·Δt; x += vx·Δt; y += vy·Δt` |
| 4. Va chạm | Khoảng cách từ tâm quả đến **đoạn thẳng** nối 2 vị trí ngón tay `≤ R` |
| 5. Trạng thái | Chém trúng: `score += 1`; bỏ lỡ: `lives -= 1`; hết mạng: Game Over |

Chi tiết xem file [`Project_Report.md`](Project_Report.md).
## 💻 Yêu cầu hệ thống

- Python **3.9 - 3.11** (khuyến nghị 3.10)
- Webcam hoạt động
- Windows / macOS / Linux
- Thư viện: `opencv-python`, `mediapipe`, `numpy` (xem `requirements.txt`)

## ⚙️ Hướng dẫn cài đặt

```bash
# 1. Tải mã nguồn
git clone https://github.com/KieuDat08/mediapipe-fruit-ninja.git
cd mediapipe-fruit-ninja

# 2. (Khuyến nghị) Tạo môi trường ảo
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy game
python main.py
```

Nếu không mở được camera, sửa `CAM_INDEX` ở đầu `main.py` (thử `1`, `2`...).

## 🎮 Hướng dẫn cách chơi

1. Ngồi cách webcam khoảng **50 - 80 cm**, đảm bảo **đủ sáng**, bàn tay nằm trọn trong khung hình.
2. Giơ **ngón trỏ** lên, bạn sẽ thấy chấm sáng ở đầu ngón tay.
3. **Vung tay nhanh** qua các quả để chém. Vệt kiếm sáng lên khi tốc độ đủ lớn.
4. Mỗi quả để rơi xuống đáy sẽ **mất 1 mạng** (có 3 mạng).
5. Hết mạng là Game Over. Phím tắt:

| Phím | Chức năng |
|---|---|
| `R` | Chơi lại |
| `Q` / `ESC` | Thoát |

**Mẹo:** nếu khó chém trúng, tăng độ sáng phòng hoặc giảm `SWIPE_SPEED_RATIO` trong `main.py`.

## 📁 Cấu trúc dự án

```
fruit_ninja/
├── main.py            # Toàn bộ mã nguồn (OOP)
├── requirements.txt
├── Project_Report.md         # Báo cáo kỹ thuật
└── README.md
```

## 🚀 Hướng phát triển

Thêm bom, combo, bộ lọc Kalman làm mượt tọa độ, âm thanh, lưu điểm cao, chơi hai tay.

## 📄 Giấy phép

Dự án phục vụ mục đích học tập.
