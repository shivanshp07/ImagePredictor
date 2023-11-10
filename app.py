from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import cv2
import requests
from io import BytesIO
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Define the image directory
IMAGE_DIR = 'static/images'

# Load a pre-trained CNN model (e.g. VGG16)
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is not None and img.size > 0:
        img = cv2.resize(img, (224, 224))  # resize the image to (224, 224)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = np.expand_dims(img, axis=0) # Add a batch dimension
    return img



def extract_features(img_path):
    img = preprocess_image(img_path)
    features = model.predict(img)
    features = np.reshape(features, (7*7*512,))
    return features



# Define a function to compute the similarity between two feature vectors using cosine similarity


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    sim = dot / norm
    return sim

# Define a function to find the most similar images to a given image in a directory


def find_similar_images(query_img_path, dir_path, top_k=5):
    query_features = extract_features(query_img_path)
    sims = []
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(dir_path, filename)
            img_features = extract_features(img_path)
            sim = cosine_similarity(query_features, img_features)
            sims.append((img_path, sim))
            print(sims)
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

# Authenticate the Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'path/to/service_account.json'
creds = None
creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=creds)

# Define the Flask app
app = Flask(__name__, template_folder='templates')

# Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Define the find similar images route
@app.route('/find_similar_images', methods=['POST'])
def find_similar_images_route():
    # Get the uploaded image from Google Drive link
    link = request.form['link']
    file_id = link.split('/')[-2]
    file = service.files().get(fileId=file_id).execute

    # Find the most similar images
    similar_images = find_similar_images('tmp.jpg', IMAGE_DIR)
    print(similar_images)

    # Render the results template with the similar images and captions
    return render_template('results.html', similar_images=similar_images)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
