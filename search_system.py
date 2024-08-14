import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image

tamanio_imagen = 224

def preprocess_image(image):
    image = tf.image.resize(image, (tamanio_imagen, tamanio_imagen))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_and_preprocess_image(image):
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image = keras_image.load_img(image, target_size=(tamanio_imagen, tamanio_imagen))
        image_array = keras_image.img_to_array(image)

    image_array = tf.image.resize(image_array, (tamanio_imagen, tamanio_imagen))
    image_array = tf.cast(image_array, tf.float32) / 255.0
    preprocessed_image = np.expand_dims(image_array, axis=0)
    return preprocessed_image, image_array

def extract_features_image(image, model):
    feature_maps = model.predict(image)
    features_flattened = feature_maps.flatten()
    return normalize([features_flattened])[0]

def search_similar_images(knn, query_feature, train_labels, k=10):
    query_feature_normalized = normalize([query_feature])[0]
    distances, indices = knn.kneighbors([query_feature_normalized], n_neighbors=k)

    first_neighbor_label = train_labels[indices[0][0]]
    filtered_indices = [index for i, index in enumerate(indices[0]) if train_labels[index] == first_neighbor_label]
    filtered_distances = [dist for i, dist in enumerate(distances[0]) if
                          train_labels[indices[0][i]] == first_neighbor_label]

    while len(filtered_indices) < k and len(filtered_distances) < len(distances[0]):
        # Añadir vecinos adicionales hasta alcanzar k vecinos
        for i in range(len(filtered_indices), k):
            if i < len(distances[0]):
                filtered_indices.append(indices[0][i])
                filtered_distances.append(distances[0][i])

    return filtered_distances[:k], filtered_indices[:k]
