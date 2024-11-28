from flask import Flask, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLO model
model = YOLO('E:\\School\\PBL\\PBL6\\model\\Server\\server\\pythonProject\\model\\weights\\v4.pt')


@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image file"}), 400
    results = model(img)
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "label": model.names[int(box.cls)],
                # "confidence": float(box.conf),
                # "box": [float(coord) for coord in box.xyxy[0]]
            })

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
