import os

from flask import Flask, jsonify, request
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import traceback

app = Flask(__name__)

model_path = 'server/pythonProject/code/v4.pt'

try:
    model = YOLO(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    exit(1)

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON received"}), 400

        if 'image' not in data:
            return jsonify({"error": "No 'image' field in JSON"}), 400

        try:
            image_data = base64.b64decode(data['image'])
            file_bytes = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as decode_error:
            return jsonify({"error": f"Error decoding base64 image: {decode_error}"}), 400

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        try:
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
        except Exception as detection_error:
            return jsonify({"error": f"Error during detection: {detection_error}"}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
