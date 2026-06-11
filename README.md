from flask import Flask, render_template_string, Response
import cv2
import numpy as np
import random
import json
from threading import Lock

app = Flask(__name__)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Camera and game variables
cap = cv2.VideoCapture(0)
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
PLAYER_SIZE = 40
OBJECT_SIZE = 20

# Game state
game_state = {
    'score': 0,
    'face_x': FRAME_WIDTH // 2,
    'face_y': FRAME_HEIGHT // 2,
    'objects': [[random.randint(0, FRAME_WIDTH), random.randint(0, FRAME_HEIGHT)] for _ in range(5)]
}
state_lock = Lock()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face-Controlled Game</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: #1a1a1a;
            font-family: Arial, sans-serif;
        }
        .container {
            width: 100%;
            max-width: 600px;
            padding: 20px;
            text-align: center;
        }
        h1 {
            color: #00ff00;
            margin-bottom: 10px;
        }
        .score {
            font-size: 24px;
            color: #00ff00;
            margin-bottom: 20px;
        }
        img {
            width: 100%;
            max-width: 500px;
            border: 3px solid #00ff00;
            border-radius: 10px;
        }
        .instructions {
            color: #ffffff;
            margin-top: 20px;
            font-size: 14px;
            line-height: 1.6;
        }
        .button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 10px 20px;
            margin-top: 10px;
            cursor: pointer;
            font-size: 16px;
            border-radius: 5px;
        }
        .button:hover {
            background: #00cc00;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎮 Face-Controlled Game</h1>
        <div class="score" id="score">Score: 0</div>
        <img src="/video_feed" alt="Camera Feed">
        <div class="instructions">
            <p>📱 Move your face to collect yellow circles</p>
            <p>👤 Your face is the blue circle</p>
            <p>⭐ Collect all yellow objects to increase score</p>
            <p>Works on Desktop & Mobile!</p>
        </div>
    </div>
    <script>
        setInterval(function() {
            fetch('/game_state')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('score').textContent = 'Score: ' + data.score;
                });
        }, 500);
    </script>
</body>
</html>
'''

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        with state_lock:
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                game_state['face_x'] = x + w // 2
                game_state['face_y'] = y + h // 2
                
                # Draw face detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw player
            cv2.circle(frame, (game_state['face_x'], game_state['face_y']), PLAYER_SIZE, (255, 0, 0), -1)
            
            # Draw and check collision
            for obj in game_state['objects']:
                cv2.circle(frame, (obj[0], obj[1]), OBJECT_SIZE, (0, 255, 255), -1)
                
                distance = np.sqrt((game_state['face_x'] - obj[0])**2 + (game_state['face_y'] - obj[1])**2)
                if distance < PLAYER_SIZE + OBJECT_SIZE:
                    game_state['score'] += 10
                    obj[0] = random.randint(0, FRAME_WIDTH)
                    obj[1] = random.randint(0, FRAME_HEIGHT)
            
            # Display info
            cv2.putText(frame, f'Score: {game_state["score"]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, 'Move your face to collect circles', (10, FRAME_HEIGHT - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game_state')
def get_game_state():
    with state_lock:
        return {'score': game_state['score']}

if __name__ == '__main__':
    print("Starting Face-Controlled Game Web Server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Access from mobile: http://<your-computer-ip>:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

