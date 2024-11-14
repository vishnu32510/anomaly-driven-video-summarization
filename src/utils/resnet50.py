import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, ReLU, MaxPooling3D, GlobalAveragePooling3D, Add, Dense
from tensorflow.keras.models import Model

# 3D Convolutional Block
def conv3d_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
    """
    A convolutional block with Conv3D, BatchNormalization, and ReLU activation.
    
    Args:
        x (tensor): Input tensor to the block.
        filters (int): Number of filters for the Conv3D layer.
        kernel_size (tuple): The kernel size for the Conv3D layer.
        strides (tuple): The strides for the Conv3D layer.
        padding (str): The padding method for Conv3D layer.
    
    Returns:
        tensor: The output tensor after applying Conv3D, BatchNorm, and ReLU.
    """
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# Residual Block
def residual_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
    """
    A residual block that adds the input to the output of the block.
    
    Args:
        x (tensor): Input tensor to the block.
        filters (int): Number of filters for the Conv3D layers.
        kernel_size (tuple): The kernel size for the Conv3D layers.
        strides (tuple): The strides for the Conv3D layers.
        padding (str): The padding method for Conv3D layers.
    
    Returns:
        tensor: The output tensor after adding the input to the output.
    """
    shortcut = x  # Save the input tensor for the residual connection
    
    # Apply 1x1x1 convolution to match the number of filters in the shortcut path
    if x.shape[-1] != filters:  # Only apply if the number of filters is different
        shortcut = Conv3D(filters, (1, 1, 1), padding='same')(shortcut)
    
    # First convolutional block
    x = conv3d_block(x, filters, kernel_size, strides, padding)
    
    # Second convolutional block
    x = conv3d_block(x, filters, kernel_size, strides, padding)
    
    # Add the shortcut (residual connection)
    x = Add()([x, shortcut])
    
    return x

# 3D ResNet Model for Feature Extraction
def build_3d_resnet_feature_extractor(input_shape=(32, 64, 64, 3)):
    """
    Build a 3D ResNet model for feature extraction.
    
    Args:
        input_shape (tuple): Shape of the input video segments (frames, height, width, channels).
        
    Returns:
        model (tf.keras.Model): The 3D ResNet feature extractor model.
    """
    inputs = Input(shape=input_shape)

    # Initial Conv3D Block
    x = conv3d_block(inputs, 64, kernel_size=(3, 3, 3), strides=(1, 1, 1))
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    # Residual Blocks
    x = residual_block(x, 128)
    x = residual_block(x, 256)
    x = residual_block(x, 512)

    # Global Average Pooling (to reduce spatial dimensions)
    x = GlobalAveragePooling3D()(x)
    
    # Final dense layer for feature extraction (optional)
    x = Dense(4096, activation='relu')(x)  # Output a 4096-dimensional feature vector

    # Define the model (this model only extracts features, no classification)
    model = Model(inputs=inputs, outputs=x, name="3D_ResNet_Feature_Extractor")
    
    return model

import numpy as np

def extract_features_3d_resnet(video_segments, resnet_model):
    """
    Extract feature vectors from a batch of video segments using the 3D ResNet model.
    
    Args:
        video_segments (np.array): A batch of video segments, each with shape (frames, height, width, channels).
        resnet_model (tf.keras.Model): The pre-trained 3D ResNet model used for feature extraction.
    
    Returns:
        np.array: An array of feature vectors extracted from the video segments.
    """
    # Preprocess video segments and extract features
    features = []
    for segment in video_segments:
        segment = np.expand_dims(segment, axis=0)  # Add batch dimension (1, frames, height, width, channels)
        feature_vector = resnet_model.predict(segment)  # Extract feature vector using the ResNet model
        features.append(feature_vector.flatten())  # Flatten the feature vector for easier handling
    return np.array(features)
