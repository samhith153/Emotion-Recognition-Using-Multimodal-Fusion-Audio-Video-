# Emotion-Recognition-Using-Multimodal-Fusion-Audio-Video-

This project implements a multimodal emotion recognition system that analyzes both video (facial expressions) and audio (voice) inputs to predict emotions in real-time. It uses deep learning models for video and audio processing, fused to provide emotion probabilities over time.

## Features

- **Real-time Capture**: Captures video from webcam and audio from microphone simultaneously.
- **Face Detection and Mesh**: Uses MediaPipe for face detection and landmark extraction.
- **Emotion Prediction**: Predicts 7 emotions: happy, sad, angry, neutral, surprise, fear, disgust.
- **Fusion**: Combines video and audio predictions for more accurate results.
- **Visualization**: Generates graphs showing emotion probabilities over time.

## Project Structure

- `src/capture.py`: Main script for capturing video and audio, preprocessing, and predicting emotions.
- `src/preprocess.py`: Functions for preprocessing video frames (face cropping, resizing) and audio (mel spectrograms).
- `src/predict.py`: Contains the neural network models for video (ResNet18) and audio (custom CNN), and prediction logic.
- `emotion_probabilities_graph.png`: Example output graph of emotion probabilities.
- `emotion_graph.png`: Another example graph.
- `happiness_graph.png`: Graph focused on happiness.

## Dependencies

Install the required packages using pip:

```
pip install opencv-python numpy sounddevice mediapipe torch torchvision librosa matplotlib
```

Note: Ensure you have a compatible audio device for sounddevice. For GPU acceleration, install PyTorch with CUDA support if available.

## Usage

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Emotion-Recognition-Using-Multimodal-Fusion-Audio-Video-
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the capture and prediction script:
   ```
   python src/capture.py
   ```

   This will:
   - Capture 10 seconds of video and audio.
   - Process the data.
   - Predict emotions and display probabilities.
   - Save a graph of emotion probabilities.

## Model Details

- **Video Model**: ResNet18 fine-tuned for emotion classification on facial images.
- **Audio Model**: Custom CNN processing mel spectrograms.
- **Fusion**: Simple averaging of probabilities from both modalities.

## Output

The script outputs:
- Console logs of emotion probabilities at each timestamp.
- A PNG graph showing emotion probabilities over time (filtered for emotions with max prob > 0.65).

## Requirements

- Python 3.7+
- Webcam and microphone for input.
- Sufficient RAM for processing (especially for video frames).

## License

This project is open-source. Feel free to use and modify.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please create an issue on GitHub.
