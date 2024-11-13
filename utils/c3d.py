import numpy as np
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten
from tensorflow.keras.models import Model

def build_c3d_feature_extractor(input_shape=(32, 64, 64, 3)):
    inputs = Input(shape=input_shape)

    # 1st Conv3D Block
    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    # 2nd Conv3D Block
    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x)

    # 3rd Conv3D Block
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # 4th Conv3D Block
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Flatten to get feature vector
    x = Flatten()(x)
    model = Model(inputs, x, name="C3D_Feature_Extractor")

    return model


def extract_features(video_segments, c3d_model):
    features = []
    for segment in video_segments:
        segment = np.expand_dims(segment, axis=0)
        feature_vector = c3d_model.predict(segment)
        features.append(feature_vector.flatten())
    return np.array(features)



