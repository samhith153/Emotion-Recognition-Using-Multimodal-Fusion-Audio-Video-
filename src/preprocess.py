# preprocess.py
import cv2
import numpy as np
import librosa
import mediapipe as mp

# =========================
# Video preprocessing
# =========================
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def preprocess_frames(frames, target_size=(224, 224)):
    preprocessed = []
    for frame, ts in frames:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections:
            # Take the first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            # Crop face
            face = frame[y:y+h, x:x+w]
            if face.size > 0:
                # Resize to target size
                face_resized = cv2.resize(face, target_size)
                # Normalize to 0-1
                face_normalized = (face_resized / 255.0).astype(np.float32)
                preprocessed.append((face_normalized, ts))
            else:
                # If no valid face, use resized full frame
                frame_resized = cv2.resize(frame, target_size)
                frame_normalized = (frame_resized / 255.0).astype(np.float32)
                preprocessed.append((frame_normalized, ts))
        else:
            # No face detected, use full frame
            frame_resized = cv2.resize(frame, target_size)
            frame_normalized = (frame_resized / 255.0).astype(np.float32)
            preprocessed.append((frame_normalized, ts))
    return preprocessed

# =========================
# Audio preprocessing
# =========================
def preprocess_audio(audio_buffer, timestamps, sr=16000, n_mels=64):
    hop_length = 512
    mel_specs = []

    chunk_size = sr
    for start in range(0, len(audio_buffer), chunk_size):
        end = start + chunk_size
        if end > len(audio_buffer):
            break
        chunk = audio_buffer[start:end]
        chunk_ts = timestamps[start] if start < len(timestamps) else 0

        mel_spec = librosa.feature.melspectrogram(
            y=chunk,
            sr=sr,
            n_mels=n_mels,
            hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db / 80.0).astype(np.float32)  # ensure float32
        mel_specs.append((mel_spec_norm, chunk_ts))
    return mel_specs
