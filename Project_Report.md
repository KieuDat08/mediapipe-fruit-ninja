# BÁO CÁO KỸ THUẬT: CHÉM HOA QUẢ BẰNG TAY (Hand-Tracking Fruit Ninja)

## 1. Ý tưởng và mục tiêu

**Ý tưởng:** Người chơi dùng ngón trỏ trước webcam như một thanh kiếm để chém các quả bay lên trên màn hình, mô phỏng trò Fruit Ninja mà không cần chuột hay bàn phím.

**Mục tiêu:**
- Áp dụng quy trình `Camera -> MediaPipe -> Landmarks -> Thuật toán tự xây dựng -> Game`.
- Tự cài đặt các thuật toán cốt lõi: đổi tọa độ, tính vận tốc chém, vật lý ném xiên, phát hiện va chạm đoạn thẳng - đường tròn.
- Đếm số quả chém trúng, quản lý mạng chơi, Game Over và chơi lại.

## 2. Input và Output

| | Mô tả |
|---|---|
| **Input** | Luồng video webcam thời gian thực, phím `R` (chơi lại), `Q`/`ESC` (thoát). |
| **Output** | Cửa sổ game: ảnh camera + hoa quả + vệt kiếm + hiệu ứng tách đôi/bắn nước + số quả đã chém + số mạng còn lại + màn hình Game Over. |

## 3. Chức năng MediaPipe sử dụng

**MediaPipe Hands** là giải pháp gồm hai mô hình học sâu chạy nối tiếp:
1. **Palm Detector:** quét toàn ảnh để tìm lòng bàn tay, trả về khung bao.
2. **Hand Landmark Model:** trên vùng bàn tay đã cắt, hồi quy **21 điểm mốc 3D**.

Khi đã phát hiện, các frame sau MediaPipe **theo dõi (tracking)** bàn tay dựa trên landmark frame trước, chỉ chạy lại Palm Detector khi mất dấu. Nhờ vậy tốc độ cao và ổn định hơn.

**Dữ liệu đầu ra của landmark số 8 (`INDEX_FINGER_TIP`):**
- `x`, `y`: tọa độ **chuẩn hóa** trong [0, 1], gốc ở góc trên trái ảnh (x chia cho chiều rộng, y chia cho chiều cao).
- `z`: độ sâu tương đối so với cổ tay (game không dùng).

Cấu hình sử dụng: `max_num_hands=1`, `model_complexity=0` (nhanh), `min_detection_confidence=0.6`, `min_tracking_confidence=0.5`.

## 4. Chi tiết thuật toán xử lý

### 4.1. Chuyển đổi tọa độ MediaPipe sang pixel

Với khung hình rộng $W$, cao $H$:

$$p_x = \lfloor x \cdot W \rfloor, \qquad p_y = \lfloor y \cdot H \rfloor$$

Giá trị $x, y$ được kẹp về [0, 1] để điểm không vượt ngoài khung. Ảnh được `cv2.flip(frame, 1)` **trước** khi đưa vào MediaPipe, nên tay phải di chuyển sang phải trên màn hình, đúng trực giác.

### 4.2. Vệt kiếm và vận tốc chém

Dùng `collections.deque(maxlen=N)` lưu $N$ vị trí gần nhất của đầu ngón trỏ. Khi thêm điểm mới, điểm cũ nhất tự bị loại. Khi vẽ, đoạn nào mới hơn thì dày hơn và sáng hơn, tạo vệt đuôi mờ dần.

Vận tốc tức thời giữa hai frame liên tiếp (đơn vị px/giây):

$$v = \frac{\sqrt{(x_t - x_{t-1})^2 + (y_t - y_{t-1})^2}}{\Delta t}$$

Dùng $\Delta t$ thực đo (`time.perf_counter`) thay vì giả định FPS cố định, nên kết quả đúng kể cả khi máy lag. Để giảm nhiễu, vận tốc được làm mượt bằng **trung bình trượt mũ (EMA)**:

$$\bar v_t = \alpha\, v_t + (1-\alpha)\,\bar v_{t-1}, \quad \alpha = 0.5$$

**Điều kiện chém:** $\bar v_t \ge V_{th}$ với $V_{th} = 0.9 \cdot H$ px/giây (tỉ lệ theo chiều cao khung hình nên không phụ thuộc độ phân giải). Nếu tay đứng yên hoặc di chuyển chậm, `slashing = False` và không có va chạm nào được tính.

### 4.3. Vật lý hoa quả (chuyển động ném xiên)

Mỗi quả có tâm $(x, y)$, bán kính $R$, vận tốc $(v_x, v_y)$ và gia tốc trọng trường $g$ (trục y hướng xuống). Cập nhật mỗi frame:

$$v_y \leftarrow v_y + g\,\Delta t, \qquad x \leftarrow x + v_x\,\Delta t, \qquad y \leftarrow y + v_y\,\Delta t$$

Kết quả là quỹ đạo **parabol**. Vận tốc ban đầu được tính từ độ cao đỉnh mong muốn $h$ (cho quả luôn bay lọt trong khung hình):

$$v_{y0} = -\sqrt{2\,g\,h}$$

Quả xuất phát dưới đáy màn hình, $v_x$ có dấu hướng về giữa màn hình để không bay ra ngoài.

### 4.4. Phát hiện va chạm đoạn thẳng - đường tròn

Nếu chỉ kiểm tra điểm đầu ngón tay hiện tại, khi vung nhanh ngón tay có thể "nhảy" qua quả giữa hai frame (hiệu ứng tunneling). Vì vậy game kiểm tra **cả đoạn thẳng** $AB$ nối vị trí frame trước ($A$) và frame hiện tại ($B$).

Khoảng cách từ tâm quả $P$ đến đoạn $AB$:

$$t = \frac{(P-A)\cdot(B-A)}{\lVert B-A\rVert^2}, \quad t \leftarrow \min(1,\max(0,t))$$

$$Q = A + t(B-A), \qquad d = \lVert P - Q\rVert = \sqrt{(P_x-Q_x)^2 + (P_y-Q_y)^2}$$

**Va chạm khi** $d \le R$. Việc kẹp $t$ về [0, 1] đảm bảo $Q$ nằm trên đoạn chứ không phải đường thẳng vô hạn. Trường hợp $A \equiv B$ thì tính trực tiếp $d = \lVert P - A\rVert$.

### 4.5. Cập nhật trạng thái và đếm điểm

```
mỗi frame:
    đọc ảnh -> lật gương -> MediaPipe -> landmark 8 -> pixel
    tính vận tốc -> slashing?
    sinh quả theo bộ đếm thời gian (càng nhiều điểm càng dày)
    với mỗi quả: cập nhật vật lý
        nếu slashing và d(quả, AB) <= R: quả.sliced = True, score += 1,
           tạo 2 nửa quả + hạt nước
    quả chưa chém mà rơi quá đáy màn hình: lives -= 1
    nếu lives <= 0: state = GAME_OVER (cập nhật điểm cao nhất)
    vẽ quả, hiệu ứng, vệt kiếm, HUD
phím R: reset | Q/ESC: thoát
```

**Hiệu ứng tách đôi:** hai nửa quả vẽ bằng `cv2.ellipse` (cung 0-180° và 180-360°), xoay theo hướng vệt chém $\theta = \operatorname{atan2}(B_y - A_y,\; B_x - A_x)$. Hai nửa nhận thêm vận tốc theo hướng vuông góc vệt chém ($\theta \pm 90^\circ$) nên tách ra hai phía, đồng thời bị trọng lực kéo xuống.

**Máy trạng thái:** `PLAYING` -> (hết mạng) -> `GAME_OVER` -> (phím R) -> `PLAYING`.

## 5. Khó khăn thường gặp và hướng phát triển

**Khó khăn:**
- **Mất dấu tay khi vung nhanh (motion blur):** ảnh mờ khiến MediaPipe không phát hiện được tay. Giải pháp trong code: xóa vệt kiếm sau vài frame mất tay để tránh nối đoạn thẳng sai giữa hai vị trí xa nhau.
- **Ánh sáng:** thiếu sáng hoặc ngược sáng làm giảm độ chính xác. Nên chơi ở nơi đủ sáng, nền không quá giống màu da.
- **Độ trễ (latency):** webcam 30 FPS cộng thời gian suy luận gây trễ. Dùng `model_complexity=0` và kiểm tra va chạm theo đoạn thẳng để bù.
- **Nhiễu landmark:** vị trí rung nhẹ giữa các frame, xử lý bằng EMA khi tính vận tốc.
- **Ngưỡng vận tốc:** quá thấp thì dễ chém nhầm, quá cao thì khó chơi; có thể chỉnh `SWIPE_SPEED_RATIO`.

**Hướng phát triển:**
- Thêm **bom** (chém trúng mất mạng hoặc kết thúc game).
- **Combo:** chém nhiều quả trong một nhát để nhân điểm.
- Thêm âm thanh, nhiều chế độ chơi (giới hạn thời gian, tăng độ khó), lưu điểm cao vào file.
- Hỗ trợ hai tay / hai người chơi (`max_num_hands=2`).
