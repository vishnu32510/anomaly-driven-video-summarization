import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model

class VideoSummarizer:
    def __init__(self):
        # Load your pre-trained models
        c3d_model_path = "/Users/vishnu/garage/cs512-f24-sellamshanmugavel-vishnupriyan/project/src/models/c3d_model.h5"  # Update with your C3D model path
        resnet_model_path = "/Users/vishnu/garage/cs512-f24-sellamshanmugavel-vishnupriyan/project/src/models/resnet3d_model.h5"  # Update with your ResNet model path
        print("Loading C3D model from:", c3d_model_path)
        self.c3d_model = load_model(c3d_model_path)
        print("C3D model loaded successfully.")
        print("Loading ResNet model from:", resnet_model_path)
        self.resnet_model = load_model(resnet_model_path)
        print("ResNet model loaded successfully.")
        
    def extract_frames(self, video_path, frame_interval=16, sequence_length=16):
        print(f"Extracting frames from video: {video_path} with interval: {frame_interval}")
        video = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                print("No more frames to read.")
                break
            if count % frame_interval == 0:
                frames.append(frame)
                print(f"Frame {count} extracted.")
            count += 1

        video.release()
        print(f"Total frames extracted: {len(frames)}")

        # Create sequences of frames
        sequences = []
        for i in range(0, len(frames) - sequence_length + 1):
            sequence = frames[i:i + sequence_length]
            sequences.append(sequence)
        
        print(f"Total sequences created: {len(sequences)}")
        return np.array(sequences)


    def predict_anomalies(self, sequences):
        print(f"Predicting anomalies for {len(sequences)} sequences.")
        predictions = []
        for i, sequence in enumerate(sequences):
            # Preprocess the sequence as needed
            processed_sequence = self.preprocess_sequence(sequence)
            print(f"Predicting sequence {i} with shape: {processed_sequence.shape}")
            
            # Predict using both models
            c3d_prediction = self.c3d_model.predict(processed_sequence)
            resnet_prediction = self.resnet_model.predict(processed_sequence)
            print(f"C3D prediction for sequence {i}: {c3d_prediction}")
            print(f"ResNet prediction for sequence {i}: {resnet_prediction}")

            # Change this line to check for anomalies
            # if np.any(c3d_prediction == 1) or np.any(resnet_prediction == 1):  # Check if any prediction indicates anomaly
            if np.any(c3d_prediction == 1):  # Check if any prediction indicates anomaly
                predictions.append(1)  # Anomaly
                print(f"Sequence {i} identified as anomaly.")
            else:
                predictions.append(0)  # Normal
                print(f"Sequence {i} identified as normal.")
        return predictions


    def preprocess_sequence(self, sequence):
        print(f"Preprocessing sequence of length: {len(sequence)}")
        processed_sequence = []
        for frame in sequence:
            # Implement your preprocessing (e.g., resizing, normalization)
            frame_resized = cv2.resize(frame, (64, 64))  # Example for C3D model
            frame_normalized = frame_resized / 255.0  # Normalization
            processed_sequence.append(frame_normalized)
        return np.expand_dims(np.array(processed_sequence), axis=0)

    def preprocess_frame(self, frame):
        # Implement your preprocessing (e.g., resizing, normalization)
        print(f"Preprocessing frame of shape: {frame.shape}")
        # Example for resizing
        frame_resized = cv2.resize(frame, (64, 64))  # Example for C3D model
        frame_normalized = frame_resized / 255.0  # Normalization
        print(f"Frame resized to: {frame_resized.shape} and normalized.")
        return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension

    def create_summary(self, input_video_path, output_video_path):
        print(f"Creating summary for video: {input_video_path} and saving to: {output_video_path}")
        frames = self.extract_frames(input_video_path)
        predictions = self.predict_anomalies(frames)

        # Open video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (64, 64))

        if not out.isOpened():
            print("Error: Could not open video writer.")
            return

        for i, frame in enumerate(frames):
            if predictions[i] == 0:
                if frame is None or len(frame) == 1:
                    print(f"Frame {i} is None or empty; skipping.")
                    continue
                
                if frame.shape[0] == 16:  # Check if frame is a batch of frames
                    for j in range(frame.shape[0]):
                        single_frame = frame[j]
                        if single_frame is None or single_frame.shape[0] == 0 or single_frame.shape[1] == 0:
                            print(f"Single frame {j} has invalid dimensions; skipping.")
                            continue
                        
                        frame_resized = cv2.resize(single_frame, (64, 64))  # Resize
                        out.write(frame_resized)
                        print(f"Writing single frame {j} from batch {i} to summary video.")
                else:
                    print(f"Processing frame {i} with shape: {frame.shape}")
                    frame_resized = cv2.resize(frame, (64, 64))  # Resize
                    out.write(frame_resized)

        out.release()
        print("Summary video creation completed.")



    def summarize_from_path(self, video_path):
        output_dir = os.path.join('summarized_videos', os.path.basename(video_path).split('.')[0])
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, 'summary.mp4')
        print(f"Summarizing video from path: {video_path} to {output_video_path}")
        self.create_summary(video_path, output_video_path)
        print("Summarization completed. Output video path:", output_video_path)
        return output_video_path
