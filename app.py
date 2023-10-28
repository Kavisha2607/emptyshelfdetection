from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("C:\\Users\\Dell\\Downloads\\Supermarket Empty Shelf "
                   "Detector.v4i.yolov8-20231027T130811Z-001\\Supermarket "
                   "Empty Shelf Detector.v4i.yolov8\\trained_model.h5")  # Replace with the actual path to your trained model file

@app.route('/')
def index():
    return render_template('index.html', result='')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Load and preprocess the image
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the trained model
        prediction = model.predict(image)
        result = 'Empty shelf detected!' if prediction < 0.5 else 'Shelf is not empty.'

        return render_template('index.html', result=result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
