# Real-Time Emotion Detection System 🎭

This project uses OpenCV, Flask, and FER to detect human emotions from webcam in real time, display quotes and emojis, and play emotion-based videos.

## Features
- Detects emotions in real time
- Displays emoji and motivational quotes
- Plays a video based on detected emotion

## Technologies Used
- Python
- Flask
- OpenCV
- FER (Facial Emotion Recognition)
- HTML/CSS

## How to Run
1. Clone this repo  
2. Run: `pip install -r requirements.txt`  
3. Start: `python app.py`  
4. Open browser at: `http://127.0.0.1:5000/`

## Folder Structure
- `/templates`: HTML files
- `/static`: CSS, images
- `/desktop_videos`: Videos to play per emotion

## Video Directory
Put videos like:
- `happy_refresh.mp4`
- `sad_refresh.mp4`
...inside a folder named `desktop_videos`.

