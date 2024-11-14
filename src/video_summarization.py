import tensorflow as tf

# Load your trained model
# model_path = 'densenet.h5'  # Update with your model's path
# model = tf.keras.models.load_model(model_path)

import cv2
import os

# model_path = '/Users/vishnu/garage/cs512-f24-sellamshanmugavel-vishnupriyan/project/src/densenet.h5'  # Update with your model's path
# model = tf.keras.models.load_model(model_path)

def extract_frames(video_path, output_dir):
    print(f"Extracting Frames")
    print(video_path,output_dir)
    print(output_dir)
    # Create a directory for frames
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame as image
        cv2.imwrite(os.path.join(output_dir, f'frame_{count:04d}.png'), frame)
        count += 1
    cap.release()
    print(f"Extracted {count} frames to {output_dir}")

# video_path = '/Users/vishnu/Desktop/car_a.mp4'  # Update with your video path
# output_dir = '/Users/vishnu/Desktop/a'    # Update with your desired output directory
# extract_frames(video_path, output_dir)

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_HEIGHT, IMG_WIDTH = 64,64

def predict_frames(frames_dir, model):
    print("Pridicting Frames")
    predictions = []
    frame_files = sorted(os.listdir(frames_dir))
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        img = load_img(frame_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        prediction = model.predict(img_array)
        predictions.append(prediction)
    print("Frames Predicted")
    return predictions

# predictions = predict_frames(output_dir)
from scipy.stats import entropy

def score_frames(predictions):
    print("Scoring Frames")
    scores = []
    for i, pred in enumerate(predictions):
        # Example scoring based on entropy of class probabilities
        score = -np.sum(pred * np.log(pred + 1e-10))  # Entropy
        # Additional scoring mechanisms can be added here
        scores.append(score)
    print("Frames Scored")
    return scores

# scores = score_frames(predictions)
# print(scores)
def rank_shots(scores):
    print("Ranking Frames")
    ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order
    print("Frames Ranked")
    return ranked_indices

# ranked_indices = rank_shots(scores)
# print(ranked_indices)
def summarize_video(frames_dir, ranked_indices, output_video_path, num_keyframes=240, frame_rate=24):
    print("Summarizing Frames")
    selected_frames = ranked_indices[:num_keyframes]
    keyframe_paths = [os.path.join(frames_dir, f'frame_{index:04d}.png') for index in selected_frames][::-1]
    
    # Check the first frame to get dimensions
    first_frame = cv2.imread(keyframe_paths[0])
    if first_frame is None:
        print(f"Error loading image: {keyframe_paths[0]}")
        return

    height, width, layers = first_frame.shape

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    for keyframe_path in keyframe_paths:
        print(f"Attempting to load: {keyframe_path}")  # Debug output
        img = cv2.imread(keyframe_path)
        if img is None:
            print(f"Error loading image: {keyframe_path}")
        else:
            # cv2.imshow("Keyframe", img)
            img_resized = cv2.resize(img, (width, height))
            out.write(img_resized)
            # cv2.waitKey(500)
    out.release()
    print(f"Summarized video saved to {output_video_path}")
    # cv2.destroyAllWindows()

# summarize_video(output_dir, ranked_indices, "input_output/output/Arson002_x264/car_a.mp4")
def summarize_from_path(video_path):
    foldername = video_path.split("/")[-1].split('.')[0]
    output_dir = f'/Users/vishnu/garage/cs512-f24-sellamshanmugavel-vishnupriyan/project/src/summarized_videos/{foldername}'
    output_dir_frames = os.path.join(output_dir,'frames')
    output_video_path = os.path.join(output_dir,'summary.mp4')
    print(foldername)
    # Load your trained model
    print(f"Start")
    model_path = '/Users/vishnu/garage/cs512-f24-sellamshanmugavel-vishnupriyan/project/src/models/densenet.h5'  # Update with your model's path
    model = tf.keras.models.load_model(model_path)
    print("Extracting Frames_1")
    extract_frames(video_path, output_dir_frames)
    predictions = predict_frames(output_dir_frames, model)
    scores = score_frames(predictions)
    ranked_indices = rank_shots(scores)
    summarize_video(output_dir_frames, ranked_indices, output_video_path)
    return output_video_path

# summarize_from_path('/Users/vishnu/Desktop/car_a.mp4')