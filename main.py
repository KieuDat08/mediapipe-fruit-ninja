"""
CHÉM HOA QUẢ BẰNG TAY (Hand-Tracking Fruit Ninja)
Quy trình: Camera -> MediaPipe Hands -> Landmark số 8 -> Thuật toán tự xây dựng -> Game

Điều khiển:  R = chơi lại | Q hoặc ESC = thoát
Lưu ý: cv2.putText không hiển thị được tiếng Việt có dấu nên chữ trên màn hình dùng tiếng Anh.
"""

import math
import random
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
# ============================================================x 
# CẤU HÌNH
# ============================================================
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
TRAIL_LENGTH = 12              # N điểm gần nhất của vệt kiếm
SWIPE_SPEED_RATIO = 0.4        # ngưỡng chém = 0.9 * chiều cao khung hình (px/giây)
SPEED_SMOOTH = 0.8             # hệ số làm mượt vận tốc (EMA)
START_LIVES = 3
HAND_LOST_FRAMES = 4           # mất tay quá số frame này thì xóa vệt kiếm

# Màu BGR của các loại quả: (tên, màu vỏ, màu ruột)
FRUIT_TYPES = [
    ("watermelon", (60, 160, 40), (80, 80, 230)),
    ("orange",     (0, 140, 255), (60, 190, 255)),
    ("apple",      (50, 50, 220), (170, 230, 250)),
    ("lemon",      (0, 230, 240), (170, 250, 250)),
    ("grape",      (170, 60, 130), (210, 150, 200)),
    ("kiwi",       (40, 130, 90), (60, 200, 120)),
]


# ============================================================
# HÀM HÌNH HỌC TỰ XÂY DỰNG
# ============================================================
def normalized_to_pixel(nx, ny, width, height):
    """
    Chuyển tọa độ chuẩn hóa của MediaPipe (x, y thuộc [0, 1]) sang pixel:
        px = x * width ,  py = y * height
    Có kẹp (clamp) để điểm không vượt ra ngoài khung hình.
    """
    px = int(min(max(nx, 0.0), 1.0) * width)
    py = int(min(max(ny, 0.0), 1.0) * height)
    return px, py


def point_segment_distance(p, a, b):
    """
    Khoảng cách ngắn nhất từ điểm p đến đoạn thẳng AB.
    Chiếu p lên đường thẳng AB: t = ((p-a).(b-a)) / |b-a|^2, kẹp t về [0, 1],
    điểm gần nhất q = a + t(b-a), khoảng cách d = |p - q| (khoảng cách Euclid).
    """
    ax, ay = a
    bx, by = b
    px, py = p
    abx, aby = bx - ax, by - ay
    len_sq = abx * abx + aby * aby
    if len_sq == 0:                      # A trùng B: đoạn suy biến thành 1 điểm
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / len_sq
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


def segment_hits_circle(a, b, center, radius):
    """Va chạm giữa đoạn thẳng AB (đường chém) và hình tròn: d(center, AB) <= R."""
    return point_segment_distance(center, a, b) <= radius


# ============================================================
# BẮT TAY BẰNG MEDIAPIPE
# ============================================================
class HandTracker:
    """Bọc MediaPipe Hands, chỉ trả về vị trí pixel đầu ngón trỏ (landmark 8)."""

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_index_tip(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # MediaPipe cần RGB
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0].landmark[
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        return normalized_to_pixel(lm.x, lm.y, w, h)

    def close(self):
        self.hands.close()


# ============================================================
# VỆT KIẾM + VẬN TỐC CHÉM
# ============================================================
class BladeTrail:
    def __init__(self, maxlen=TRAIL_LENGTH):
        self.points = deque(maxlen=maxlen)   # N điểm gần nhất
        self.speed = 0.0                      # px/giây (đã làm mượt)
        self.prev_point = None
        self.lost_count = 0

    def update(self, point, dt, swipe_threshold):
        """Cập nhật vị trí mới; trả về (đang chém, điểm trước, điểm hiệntại)."""
        if point is None:
            self.lost_count += 1
            self.speed = 0.0
            if self.lost_count > HAND_LOST_FRAMES:
                self.points.clear()
                self.prev_point = None
            return False, None, None

        self.lost_count = 0
        prev = self.prev_point
        if prev is not None and dt > 0:
            # Vận tốc tức thời v = d / dt, với d là khoảng cách Euclid giữa 2 frame
            dist = math.hypot(point[0] - prev[0], point[1] - prev[1])
            inst_speed = dist / dt
            # Làm mượt bằng trung bình trượt mũ để tránh nhiễu nhảy số
            self.speed = SPEED_SMOOTH * inst_speed + (1 - SPEED_SMOOTH) * self.speed
        else:
            self.speed = 0.0

        self.points.append(point)
        self.prev_point = point
        # Chỉ tính là "chém" khi vận tốc vượt ngưỡng => để yên tay không ăn điểm
        slashing = prev is not None and self.speed >= swipe_threshold
        return slashing, prev, point

    def draw(self, frame, slashing):
        pts = list(self.points)
        n = len(pts)
        for i in range(1, n):
            ratio = i / n                       # đoạn càng mới càng dày, càng sáng
            thickness = max(1, int(2 + 12 * ratio))
            if slashing:
                color = (255, int(200 + 55 * ratio), int(100 + 155 * ratio))
            else:
                color = (int(120 * ratio), int(120 * ratio), int(120 * ratio))
            cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
        if n:
            cv2.circle(frame, pts[-1], 9, (0, 255, 255) if slashing else (200, 200, 200), -1)


# ============================================================
# HOA QUẢ + VẬT LÝ
# ============================================================
class Fruit:
    def __init__(self, width, height):
        self.name, self.skin, self.flesh = random.choice(FRUIT_TYPES)
        self.r = int(height * random.uniform(0.05, 0.075))
        self.g = 1.6 * height                              # gia tốc trọng trường (px/s^2)
        self.x = random.uniform(0.15 * width, 0.85 * width)
        self.y = float(height + self.r)                    # xuất phát dưới đáy khung hình
        # Chọn độ cao đỉnh quỹ đạo h, suy ra vy từ v^2 = 2*g*h (vy âm = bay lên)
        peak = random.uniform(0.55, 0.85) * height
        self.vy = -math.sqrt(2 * self.g * peak)
        # Ném hơi hướng về giữa màn hình để quả không bay ra ngoài
        direction = 1 if self.x < width / 2 else -1
        self.vx = direction * random.uniform(0.02, 0.22) * width
        self.sliced = False

    def update(self, dt):
        # Chuyển động ném xiên: vy += g*dt ; x += vx*dt ; y += vy*dt  -> quỹ đạo parabol
        self.vy += self.g * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def is_missed(self, height):
        return self.y - self.r > height and self.vy > 0

    @property
    def center(self):
        return (self.x, self.y)

    def draw(self, frame):
        c = (int(self.x), int(self.y))
        cv2.circle(frame, c, self.r, self.skin, -1, cv2.LINE_AA)
        cv2.circle(frame, c, self.r, (30, 30, 30), 2, cv2.LINE_AA)
        hl = (c[0] - self.r // 3, c[1] - self.r // 3)       # đốm sáng tạo cảm giác 3D
        cv2.circle(frame, hl, max(2, self.r // 6), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(frame, (c[0], c[1] - self.r), (c[0] + 4, c[1] - self.r - 10),
                 (30, 100, 30), 3, cv2.LINE_AA)               # cuống quả


class SlicedHalf:

    def __init__(self, fruit, slice_angle_deg, side, blade_speed):
        self.x, self.y = fruit.x, fruit.y
        self.r = fruit.r
        self.skin, self.flesh = fruit.skin, fruit.flesh
        self.angle = slice_angle_deg
        self.side = side                                    # 0 hoặc 1 (nửa trên/dưới)
        normal = math.radians(slice_angle_deg + 90)
        push = (1 if side == 0 else -1) * min(350.0, 0.2 * blade_speed + 120)
        self.vx = fruit.vx + push * math.cos(normal)
        self.vy = fruit.vy + push * math.sin(normal)
        self.g = fruit.g
        self.spin = (1 if side == 0 else -1) * random.uniform(60, 180)
        self.life = 1.0

    def update(self, dt):
        self.vy += self.g * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt
        self.life -= dt * 0.9

    def draw(self, frame):
        c = (int(self.x), int(self.y))
        start, end = (0, 180) if self.side == 0 else (180, 360)
        cv2.ellipse(frame, c, (self.r, self.r), self.angle, start, end, self.skin, -1, cv2.LINE_AA)
        cv2.ellipse(frame, c, (int(self.r * 0.82),) * 2, self.angle, start, end,
                    self.flesh, -1, cv2.LINE_AA)


class Particle:
    """Hiệu ứng nước quả bắn tung tóe."""

    def __init__(self, x, y, color):
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(150, 550)
        self.x, self.y = x, y
        self.vx, self.vy = spd * math.cos(ang), spd * math.sin(ang)
        self.color = color
        self.life = random.uniform(0.3, 0.7)
        self.size = random.randint(2, 6)

    def update(self, dt, g):
        self.vy += g * 0.5 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

    def draw(self, frame):
        cv2.circle(frame, (int(self.x), int(self.y)), self.size, self.color, -1)


# ============================================================
# GAME
# ============================================================
class Game:
    PLAYING, GAME_OVER = "PLAYING", "GAME_OVER"

    def __init__(self, width, height):
        self.w, self.h = width, height
        self.swipe_threshold = SWIPE_SPEED_RATIO * height
        self.best = 0
        self.reset()

    def reset(self):
        self.state = Game.PLAYING
        self.score = 0          # bộ đếm số quả đã chém
        self.lives = START_LIVES
        self.fruits, self.halves, self.particles = [], [], []
        self.spawn_timer = 0.0
        self.flash_text = None

    def spawn_interval(self):
        # Càng nhiều điểm càng ra quả dày hơn (tối thiểu 0.45 giây)
        return max(0.45, 1.3 - 0.03 * self.score) * random.uniform(0.7, 1.2)

    def slice_fruit(self, fruit, a, b, blade_speed):
        fruit.sliced = True
        self.score += 1
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))   # hướng vệt chém
        self.halves.append(SlicedHalf(fruit, angle, 0, blade_speed))
        self.halves.append(SlicedHalf(fruit, angle, 1, blade_speed))
        for _ in range(14):
            self.particles.append(Particle(fruit.x, fruit.y, fruit.flesh))

    def update(self, dt, slashing, a, b, blade_speed):
        # Hiệu ứng luôn được cập nhật kể cả khi Game Over
        for obj in self.halves:
            obj.update(dt)
        for p in self.particles:
            p.update(dt, 1.6 * self.h)
        self.halves = [o for o in self.halves if o.life > 0 and o.y < self.h + 100]
        self.particles = [p for p in self.particles if p.life > 0]

        if self.state != Game.PLAYING:
            return

        self.spawn_timer -= dt
        if self.spawn_timer <= 0:
            for _ in range(random.choice([1, 1, 1, 2])):
                self.fruits.append(Fruit(self.w, self.h))
            self.spawn_timer = self.spawn_interval()

        for f in self.fruits:
            f.update(dt)
            # Va chạm chỉ xét khi đang chém nhanh: đoạn a->b (2 vị trí ngón tay liên tiếp)
            if slashing and a is not None and b is not None:
                if segment_hits_circle(a, b, f.center, f.r):
                    self.slice_fruit(f, a, b, blade_speed)

        survivors = []
        for f in self.fruits:
            if f.sliced:
                continue
            if f.is_missed(self.h):
                self.lives -= 1                              # bỏ lỡ => mất mạng
                continue
            survivors.append(f)
        self.fruits = survivors

        if self.lives <= 0:
            self.state = Game.GAME_OVER
            self.best = max(self.best, self.score)

    def draw(self, frame):
        for f in self.fruits:
            f.draw(frame)
        for hf in self.halves:
            hf.draw(frame)
        for p in self.particles:
            p.draw(frame)
        self.draw_hud(frame)
        if self.state == Game.GAME_OVER:
            self.draw_game_over(frame)

    def draw_hud(self, frame):
        cv2.putText(frame, f"Sliced: {self.score}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, f"Sliced: {self.score}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)
        for i in range(START_LIVES):                          # mỗi mạng là một trái tim
            color = (60, 60, 255) if i < self.lives else (90, 90, 90)
            cx = self.w - 45 - i * 55
            cv2.circle(frame, (cx - 9, 38), 11, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx + 9, 38), 11, color, -1, cv2.LINE_AA)
            cv2.fillPoly(frame, [np.array([(cx - 19, 44), (cx + 19, 44), (cx, 68)])], color)

    def draw_game_over(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        lines = [("GAME OVER", 2.8, (60, 60, 255), 6),
                 (f"Fruits sliced: {self.score}", 1.5, (255, 255, 255), 3),
                 (f"Best: {self.best}", 1.2, (0, 220, 255), 2),
                 ("Press R to restart  |  Q to quit", 1.0, (200, 200, 200), 2)]
        y = self.h // 2 - 100
        for text, scale, color, th in lines:
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0]
            cv2.putText(frame, text, ((self.w - size[0]) // 2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)
            y += int(size[1] + 45)


# ============================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Khong mo duoc webcam. Hay kiem tra CAM_INDEX hoac quyen truy cap camera.")
        return

    tracker = HandTracker()
    trail = BladeTrail()
    game = None
    last_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                 # lật ngang tạo hiệu ứng gương
        h, w = frame.shape[:2]
        if game is None:
            game = Game(w, h)

        now = time.perf_counter()
        dt = min(now - last_time, 0.05)            # kẹp dt để vật lý không "nhảy" khi lag
        last_time = now

        tip = tracker.get_index_tip(frame)         # landmark 8 -> pixel
        slashing, a, b = trail.update(tip, dt, game.swipe_threshold)

        game.update(dt, slashing, a, b, trail.speed)
        game.draw(frame)
        trail.draw(frame, slashing)

        cv2.imshow("Hand Fruit Ninja", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            game.reset()

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
