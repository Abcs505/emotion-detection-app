from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
from fer import FER
import random
import atexit
import os
import numpy as np

app = Flask(__name__)
detector = FER(mtcnn=True)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your actual video folder on the desktop
VIDEOS_DIR = r"C:\Users\arast\OneDrive\Desktop\videos"

# Emotion quotes
quotes = {
    "happy": ["Keep smiling!", "Stay positive!"],
    "sad": ["It's okay to feel sad.", "Better days are coming."],
    "angry": ["Take a deep breath.", "Peace begins with a smile."],
    "surprise": ["Wow!", "Didn't expect that!"],
    "neutral": ["Keep going.", "Steady and strong."],
    "fear": ["Fear is temporary. Courage is forever."],
    "disgust": ["Let go of negativity.", "Focus on the good."]
}

# Emojis by emotion
emoji_dict = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "😖"
}

# Shared state for emotion
latest_data = {"emotion": "neutral"}

# Webcam
cap = cv2.VideoCapture(0)

def release_camera():
    if cap.isOpened():
        cap.release()

atexit.register(release_camera)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = detector.detect_emotions(frame)
        for face in results:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)

            # Update emotion state
            latest_data["emotion"] = dominant_emotion

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{dominant_emotion.capitalize()}"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/quote')
def get_quote():
    emotion = latest_data["emotion"]
    quote = random.choice(quotes.get(emotion, ["Stay strong!"]))
    emoji = emoji_dict.get(emotion, "☺️")
    return jsonify({"quote": quote, "emoji": emoji, "emotion": emotion})

@app.route('/desktop_videos/<path:filename>')
def desktop_videos(filename):
    filepath = os.path.join(VIDEOS_DIR, filename)
    if os.path.isfile(filepath):
        return send_from_directory(VIDEOS_DIR, filename)
    else:
        return "File not found", 404

@app.route('/play_video/<emotion>')
def play_video(emotion):
    if emotion not in quotes:
        emotion = "neutral"
    video_filename = f"{emotion}_refresh.mp4"
    video_path = f"/desktop_videos/{video_filename}"
    return render_template('video_player.html', video_path=video_path, emotion=emotion)

if __name__ == "__main__":
    app.run(debug=True)
