from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model('./Training/tomato_disease_detector.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the class indices (update this based on your actual classes)
class_indices = {
    0: "Bacterial Spot",
    1: "Early Blight",
    2: "Late Blight",
    3: "Leaf Mold",
    4: "Septoria Leaf Spot",
    5: "Spider Mites",
    6: "Target Spot",
    7: "Tomato Yellow Leaf Curl Virus",
    8: "Tomato Mosaic Virus",
    9: "Healthy"
}

def predict_disease(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        print(f"Image shape after preprocessing: {img_array.shape}")
        
        prediction = model.predict(img_array)
        print(f"Raw model prediction: {prediction}")

        predicted_class = np.argmax(prediction, axis=1)[0]
        disease_name = class_indices[predicted_class]
        return disease_name
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction Error"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html', _anchor='about')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', _anchor='predict', result='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', _anchor='predict', result='No selected file')
        if file:
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            disease_name = predict_disease(file_path)
            os.remove(file_path)  # Remove the file after prediction
            return render_template('index.html', _anchor='predict', result=disease_name)
    return render_template('index.html', _anchor='predict')

@app.route('/contact')
def contact():
    return render_template('index.html', _anchor='contact')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=3000)
