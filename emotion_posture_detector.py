import cv2
import mediapipe as mp
import math
import numpy as np
import os
import tkinter as tk
import sys
import time
from tkinter import ttk, messagebox
from tensorflow.keras.models import load_model
from pygrabber.dshow_graph import FilterGraph
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# Setup đường dẫn tới các file model, cascade và icon
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

icon_path = os.path.join(BASE_DIR, "emotion_posture_detector.ico")
face_xml = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
model_h5 = os.path.join(BASE_DIR, "emotion_detection.h5")
font_path = os.path.join(BASE_DIR, "ARIALBD 1.ttf")

# Load cascade và model
face_classifier = cv2.CascadeClassifier(face_xml)
if face_classifier.empty():
    raise FileNotFoundError(f"Không tìm thấy file cascade: {face_xml}")

classifier = load_model(model_h5)
class_labels = ['Giận dữ', 'Ghê sợ', 'Sợ hãi', 'Vui vẻ', 'Buồn', 'Bất ngờ', 'Trung lập']

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Hàm tính góc
def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    angle = math.degrees(
        math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    )
    return abs(angle)

# Liệt kê camera
def list_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    return devices

# Vẽ chữ có viền
def draw_text_with_outline(draw, pos, text, font, text_color,
                           outline_color=(0, 0, 0), outline_width=1):
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=text_color)

# Vẽ hình vuông có viền
def draw_filled_rectangle_with_outline(img, pt1, pt2, color,
                                       outline_color=(0, 0, 0),
                                       outline_width=1):
    cv2.rectangle(img,
                  (pt1[0] - outline_width, pt1[1] - outline_width),
                  (pt2[0] + outline_width, pt2[1] + outline_width),
                  outline_color, -1)
    cv2.rectangle(img, pt1, pt2, color, -1)

# Hiện cảnh báo nhắc nhở
def show_warning(msg):
    win = tk.Toplevel()
    win.withdraw()
    win.attributes('-topmost', True)
    messagebox.showwarning("Cảnh báo", msg, parent=win)
    win.destroy()

# Chạy nhận diện
def run_detection(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở camera.")
        return

    history = deque(maxlen=150)
    start_time = time.time()
    interval = 120

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # PIL để vẽ
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        font = ImageFont.truetype(font_path, 30)

        # Nhận diện cảm xúc
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        labels = []
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (242, 248, 68), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi, verbose=0)[0]
            label = class_labels[preds.argmax()]
            labels.append(label)

            # Vẽ label bằng PIL
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            font = ImageFont.truetype(font_path, 28)

            draw_text_with_outline(draw, (x, y - 35), label, font,
                                   text_color=(0, 255, 0),
                                   outline_color=(0, 0, 0),
                                   outline_width=1)
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Lưu lịch sử
        if labels:
            if labels.count("Trung lập") >= len(labels) / 2:
                history.append(1)
            else:
                history.append(0)

        neutral_ratio = sum(history) / len(history) if len(history) > 0 else 0

        # Kiểm tra sau mỗi 2 phút
        elapsed = time.time() - start_time
        if elapsed >= interval:
            if neutral_ratio > 0.6:
                show_warning("Pause / đổi hoạt động / nghỉ 2 phút")
            start_time = time.time()

        # Nhận diện tư thế
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        status = "Đang phân tích..."
        color = (255, 255, 255)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            angle = calculate_angle(ear, shoulder, hip)

            if angle > 150:
                status = "Ngồi thẳng"
                color = (0, 255, 0)
            elif 120 < angle <= 150:
                status = "Ngồi hơi cúi"
                color = (255, 255, 0)
            else:
                status = "Ngồi cúi nhiều"
                color = (255, 0, 0)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Chọn màu ô
        if neutral_ratio > 0.6:
            box_color = (0, 0, 255)
        elif 0.2 <= neutral_ratio <= 0.6:
            box_color = (0, 255, 255)
        else:
            box_color = (0, 255, 0)

        # Vẽ thông tin
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        font = ImageFont.truetype(font_path, 30)

        draw_text_with_outline(draw, (30, 30), status, font, color,
                               outline_color=(0, 0, 0), outline_width=1)
        draw_text_with_outline(draw, (30, 75), f"Số lượng: {len(faces)}", font,
                               (255, 0, 255), outline_color=(0, 0, 0), outline_width=1)
        draw_text_with_outline(draw, (30, 120), "Trạng thái:", font,
                               (0, 0, 255), outline_color=(0, 0, 0), outline_width=1)
        draw_text_with_outline(draw, (930, 670), "Bấm phím 'Q' để thoát", font,
                               (255, 255, 0), outline_color=(0, 0, 0), outline_width=1)

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Vẽ ô trạng thái
        draw_filled_rectangle_with_outline(frame, (200, 120), (240, 160),
                                           box_color, outline_color=(0, 0, 0), outline_width=2)

        cv2.imshow('Emotion + Posture Detector', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI chọn camera
def open_camera():
    selected = combo.current()
    if selected == -1:
        messagebox.showwarning("Chưa chọn", "Vui lòng chọn một camera.")
        return
    run_detection(selected)

root = tk.Tk()
root.title("Emotion + Posture Detector")
root.attributes('-topmost', True)
root.update()
root.attributes('-topmost', False)

if os.path.exists(icon_path):
    root.iconbitmap(icon_path)

cameras = list_cameras()

label = tk.Label(root, text="Chọn camera:")
label.pack(pady=5)

combo = ttk.Combobox(root, values=cameras, state="readonly", width=50)
combo.pack(pady=5)

btn = tk.Button(root, text="Mở Camera", command=open_camera)
btn.pack(pady=10)

root.mainloop()
