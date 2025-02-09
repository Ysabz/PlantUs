from flask import Flask, render_template, jsonify, request
from markupsafe import Markup
from model import predict_image
import utils
import numpy as np
from ultralytics import YOLO
import cv2


# Load the YOLO model
yolo_model = YOLO('./Models/pest.pt')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if not file:
                return render_template('index.html', status=400, res="No file uploaded")

            # Convert file to OpenCV image
            img = file.read()
            file_bytes = np.frombuffer(img, np.uint8)
            img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Run YOLO detection
            results = yolo_model(img_cv)
            # Find the class with the highest confidence
            best_class = None
            highest_conf = 0.0

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])  # Confidence score
                    cls_id = int(box.cls[0])   # Class ID
                    class_name = yolo_model.names[cls_id]  # Get class name

                    if conf > highest_conf:  # Update if confidence is higher
                        highest_conf = conf
                        best_class = class_name
            
            res = Markup(utils.pest_dic[best_class])
            # detected_text = f"Most likely: {best_class} (Confidence: {highest_conf:.2f})" if best_class else "No objects detected"
            # Return detected classes as text
            return render_template('display.html', status=200, result=res)

        except Exception as e:
            print(f"Error: {e}")
            try:
                # Fallback prediction method
                prediction = predict_image(img)
                print(prediction)
                res = Markup(utils.disease_dic[prediction])
                return render_template('display.html', status=200, result=res)
            except Exception as fallback_error:
                print(f"Fallback Error: {fallback_error}")
                return render_template('index.html', status=500, res="Internal Server Error")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
