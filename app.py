import io
from io import BytesIO
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok

public_url = ngrok.connect(5000)
print('Public URL:', public_url)

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


app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

tamanio_imagen = 224
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                         input_shape=(tamanio_imagen, tamanio_imagen, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
train_labels = np.load('train_labels.npy', allow_pickle=True)
train_images = np.load('train_images.npy', allow_pickle=True)
train_features = np.load('train_features.npy', allow_pickle=True)
knn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(train_features)

os.makedirs('static/results/', exist_ok=True)

@app.route('/api/v1/test', methods=['GET'])
def test():
    return jsonify({'message': 'Server is working'})

@app.route('/api/v1/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')

    img_preprocessed, _ = load_and_preprocess_image(image)

    query_features = extract_features_image(img_preprocessed, model)
    distances, indices = search_similar_images(knn, query_features, train_labels, k=10)

    results = []
    for i in range(len(distances)):
        idx = indices[i]
        img_result_pil = Image.fromarray((train_images[idx] * 255).astype(np.uint8))
        img_path = f'static/results/{idx}.jpg'
        img_result_pil.save(img_path)

        results.append({
            'distance': float(distances[i]),
            'label': int(train_labels[idx]),
            'image_url': f'/static/results/{idx}.jpg'
        })

    return jsonify(results)

@app.route('/')
def index():
    return render_template('index.html')

# Ruta genérica para servir cualquier archivo desde la carpeta actual
@app.route('/<path:filename>')
def static_files(filename):
    # Utiliza el directorio actual como base
    return send_from_directory(os.getcwd(), filename)

if __name__ == '__main__':
    app = create_app()
    # Start Ngrok tunnel before running the Flask app
    public_url = run_with_ngrok(app)
    print(f"Public URL: {public_url}")
    app.run()
