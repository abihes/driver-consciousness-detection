# finallllll

import cv2
import numpy as np
import time
import threading
import requests
from tensorflow.keras.models import load_model
import geocoder
from geopy.geocoders import Nominatim
import pygame

# ---------------- Parameters ----------------
IMG_SIZE = 224
CATEGORIES = ['Open', 'Closed']  # Must match training order
MODEL_PATH = "best_model_finetuned.h5"
ALARM_FILE = "beep-warning-6387.mp3"

# Telegram Config
BOT_TOKEN = "7164066870:AAF2uGmWFlWqgzyEnYU26w9x03AR-Bg_gWk"
CHAT_ID = "1352543529"

# Toggle interpretation for sigmoid output
SIGMOID_IS_OPEN = True  # If True, sigmoid output = probability(Open); else probability(Closed)

# ---------------- Load Model ----------------
model = load_model(MODEL_PATH)

# ---------------- Initialize Pygame ----------------
pygame.mixer.init()

def play_beep_loop():
    """Play MP3 alarm in loop until stopped."""
    try:
        pygame.mixer.music.load(ALARM_FILE)
        pygame.mixer.music.play(loops=-1)  # infinite loop
    except Exception as e:
        print("Error playing beep:", e)

def stop_beep():
    """Stop the looping alarm."""
    pygame.mixer.music.stop()

# ---------------- Telegram Alert ----------------
def send_to_telegram(message: str, lat=None, lng=None):
    try:
        url_msg = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload_msg = {"chat_id": CHAT_ID, "text": message}
        r1 = requests.post(url_msg, data=payload_msg, timeout=5)

        if lat is not None and lng is not None:
            url_loc = f"https://api.telegram.org/bot{BOT_TOKEN}/sendLocation"
            payload_loc = {"chat_id": CHAT_ID, "latitude": lat, "longitude": lng}
            r2 = requests.post(url_loc, data=payload_loc, timeout=5)
    except Exception as e:
        print("Error sending Telegram:", e)

# ---------------- Location Fetch ----------------
def get_current_location():
    try:
        g = geocoder.ip('me')
        lat, lng = g.latlng
        geolocator = Nominatim(user_agent="driver_monitoring_app")
        location = geolocator.reverse(f"{lat},{lng}", language="en")
        address = location.address if location else "Unknown Address"
        return lat, lng, address
    except:
        return None, None, "Unknown"

# ---------------- Prediction ----------------
def predict_eye(eye_img):
    """Predict Open/Closed on a single eye image."""
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    eye_img = cv2.resize(eye_img, (IMG_SIZE, IMG_SIZE))
    eye_img = eye_img / 255.0
    eye_img = np.expand_dims(eye_img, axis=0)

    pred = model.predict(eye_img, verbose=0)

    if pred.shape[1] == 1:  # sigmoid
        prob = pred[0][0]
        if SIGMOID_IS_OPEN:
            is_open = prob > 0.5
            label = "Open" if is_open else "Closed"
            confidence = float(prob if is_open else 1 - prob)
        else:
            is_closed = prob > 0.5
            label = "Closed" if is_closed else "Open"
            confidence = float(prob if is_closed else 1 - prob)
    else:  # softmax
        class_index = np.argmax(pred)
        label = CATEGORIES[class_index]
        confidence = float(np.max(pred))

    return label, confidence

# ---------------- Main Monitoring Loop ----------------
def monitor_driver():
    cap = cv2.VideoCapture(0)
    start_time = None
    beep_played = False
    telegram_sent = False

    smooth_window = []
    SMOOTH_LEN = 3

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        label = "Open"
        conf = 1.0

        eye_preds = []

        for (x, y, w, h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes[:2]:
                margin = int(0.2 * ew)
                x1 = max(0, ex - margin)
                y1 = max(0, ey - margin)
                x2 = min(roi_color.shape[1], ex + ew + margin)
                y2 = min(roi_color.shape[0], ey + eh + margin)
                eye_crop = roi_color[y1:y2, x1:x2]
                if eye_crop.size == 0:
                    continue
                lbl, conf_eye = predict_eye(eye_crop)
                eye_preds.append((lbl, conf_eye))
            break

        if eye_preds:
            open_count = sum(1 for l, _ in eye_preds if l == "Open")
            closed_count = sum(1 for l, _ in eye_preds if l == "Closed")
            label = "Closed" if closed_count > open_count else "Open"
            conf = np.mean([c for _, c in eye_preds])

        smooth_window.append(label)
        if len(smooth_window) > SMOOTH_LEN:
            smooth_window.pop(0)
        label_smoothed = max(set(smooth_window), key=smooth_window.count)

        # ---------------- Drowsiness Logic ----------------
        if label_smoothed == "Closed":
            if start_time is None:
                start_time = time.time()
            elapsed = time.time() - start_time
        
            # Start looping beep after 4s
            if elapsed >= 4 and not beep_played:
                print("🔊 Eyes Closed: Starting looping beep alarm...")
                threading.Thread(target=play_beep_loop, daemon=True).start()
                beep_played = True
        
            # Send Telegram after 15s
            if elapsed >= 15 and not telegram_sent:
                print("⚠ ALERT: Driver drowsy for 15+ seconds!")
                lat, lng, address = get_current_location()
                message = (
                    f"🚨 DRIVER DROWSINESS ALERT 🚨\n\n"
                    f"Status: {label_smoothed}\n"
                    f"Latitude: {lat}\nLongitude: {lng}\n"
                    f"Google Maps: https://maps.google.com/?q={lat},{lng}\n\n"
                    f"📍 Address: {address}"
                )
                send_to_telegram(message)
                telegram_sent = True
        
        else:
            start_time = None
            if beep_played:
                stop_beep()   # 🔥 stops only when eyes open
            beep_played = False
            telegram_sent = False


        cv2.putText(frame, f"{label_smoothed} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Driver Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- Run ----------------
if __name__ == "__main__":
    monitor_driver()