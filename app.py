import io
from io import BytesIO
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from search_system import load_and_preprocess_image, extract_features_image, search_similar_images, preprocess_image
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from PIL import Image

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

tamanio_imagen = 224
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                         input_shape=(tamanio_imagen, tamanio_imagen, 3))
model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
train_features = np.load('train_features.npy')
train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
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
    app.run(debug=True)