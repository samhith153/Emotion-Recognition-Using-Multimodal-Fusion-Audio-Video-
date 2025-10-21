# capture_and_predict.py
import cv2
import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import mediapipe as mp
from collections import deque
from preprocess import preprocess_frames, preprocess_audio
from predict import predict_emotions, emotion_labels

# =========================
# Helper Classes
# =========================
class RollingFrames:
    def __init__(self, max_frames=10000):
        self.frames = []
    def add(self, frame):
        self.frames.append((frame, time.time()))
    def get(self):
        return self.frames

class RollingAudio:
    def __init__(self, sr=16000):
        self.sr = sr
        self.buffer = []
        self.timestamps = []

    def add_chunk(self, chunk):
        self.buffer.extend(chunk)
        self.timestamps.extend([time.time()] * len(chunk))

    def get(self):
        return np.array(self.buffer, dtype=np.float32), self.timestamps  # force float32

# =========================
# Audio capture
# =========================
ra = RollingAudio()
def audio_callback(indata, frames, time_info, status):
    ra.add_chunk(indata[:,0])
stream = sd.InputStream(samplerate=ra.sr, channels=1, callback=audio_callback)
stream.start()

# =========================
# Video capture
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

rf = RollingFrames()
cap = cv2.VideoCapture(0)
duration = 10
start_time = time.time()

print("Capturing video + audio...")
while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        continue
    # Process frame for face mesh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    cv2.imshow('Face Mesh', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    rf.add(frame)

cap.release()
cv2.destroyAllWindows()
stream.stop()

print(f"Captured frames: {len(rf.get())}")
audio_buffer, audio_ts = ra.get()
print(f"Captured audio samples: {len(audio_buffer)}")

# =========================
# Preprocessing
# =========================
video_input = preprocess_frames(rf.get())
audio_input = preprocess_audio(audio_buffer, audio_ts)

print(f"Preprocessed video frames: {len(video_input)}")
print(f"Preprocessed audio segments: {len(audio_input)}")

# =========================
# Prediction
# =========================
timeline = predict_emotions(video_input, audio_input, fusion='average')

print("\n=== Emotion Probabilities Timeline ===")
for ts, probs in timeline:
    print(f"Time: {ts:.2f}, Probabilities: {dict(zip(emotion_labels, [f'{p:.2f}' for p in probs]))}")

# Plot the emotion probabilities timeline (only emotions with max prob > 0.5)
times = [ts for ts, probs in timeline]
probs_list = [probs for ts, probs in timeline]

# Find emotions with max probability > 0.65
emotions_to_plot = []
for i, label in enumerate(emotion_labels):
    max_prob = max(p[i] for p in probs_list)
    if max_prob > 0.65:
        emotions_to_plot.append((i, label))

plt.figure(figsize=(12, 6))
for i, label in emotions_to_plot:
    plt.plot(times, [p[i] for p in probs_list], marker='o', linestyle='-', label=label)
plt.xlabel('Time (seconds)')
plt.ylabel('Probability')
plt.title('Emotion Probabilities Over Time (Filtered >0.65)')
plt.legend()
plt.grid(True)
plt.savefig('emotion_probabilities_graph.png')
print("Graph saved as emotion_probabilities_graph.png")
plt.show()
