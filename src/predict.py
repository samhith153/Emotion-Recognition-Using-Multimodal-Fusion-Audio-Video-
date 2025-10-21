# predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Video Model
# =========================
model_video = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_video.fc = nn.Linear(model_video.fc.in_features, 7)
# Assume model weights are Double (float64) due to persistent error
model_video.to(device)
model_video.eval()

video_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# Audio Model
# =========================
class AudioEmotionNet(nn.Module):
    def __init__(self, n_emotions=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, n_emotions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.softmax(self.fc(x), dim=1)
        return x

model_audio = AudioEmotionNet().to(device)
model_audio.eval()

# =========================
# Labels
# =========================
emotion_labels = ["happy","sad","angry","neutral","surprise","fear","disgust"]

# =========================
# Prediction
# =========================
def predict_emotions(video_input, audio_input, fusion='average'):
    timeline = []

    video_preds = []
    for frame, ts in video_input:
        # 🛠️ CRITICAL FIX: Cast input tensor to float32 to match the model's required type
        # The frame is np.float32 from preprocessing, and converted to torch.float32.
        frame_tensor = torch.tensor(frame.transpose(2,0,1), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model_video(frame_tensor)
        video_preds.append((pred.cpu(), ts))

    audio_preds = []
    for mel_spec, ts in audio_input:
        # Audio input remains float32 (standard for most audio models)
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model_audio(mel_tensor)
        audio_preds.append((pred.cpu(), ts))

    for v_pred, v_ts in video_preds:
        if audio_preds:
            closest_audio = min(audio_preds, key=lambda x: abs(x[1]-v_ts))
            a_pred, _ = closest_audio
            # Fuse predictions by adding probabilities
            combined_pred = v_pred + a_pred
        else:
            combined_pred = v_pred
        # Apply sigmoid to each probability
        sigmoid_probs = torch.sigmoid(combined_pred).squeeze().tolist()
        timeline.append((v_ts, sigmoid_probs))

    return timeline
