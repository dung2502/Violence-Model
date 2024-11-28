from flask import Flask, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

model = YOLO('E:/School/PBL/PBL6/server/pythonProject/detect/train/weights/v4.pt')

@app.route('/detect', methods=['GET'])
def detect_objects():
    image_path = 'E:\School\PBL\PBL6\server\pythonProject\image\Bang-Nhom-Giang-Ho-2.jpg'
    if not os.path.exists(image_path):
        return jsonify({"error": "Image file not found"}), 404
    img = cv2.imread(image_path)
    results = model(img)
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "label": model.names[int(box.cls)],  # Lấy nhãn từ model
                "confidence": float(box.conf),
                "box": [float(coord) for coord in box.xyxy[0]]
            })

    return jsonify({"detections": detections})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
