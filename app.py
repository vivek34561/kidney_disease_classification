from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set environment variables for locale
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Home route
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


# Route for training the model
@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


# Route for making predictions
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        filename = "inputImage.jpg"
        decodeImage(image, filename)

        # Load the model when needed
        prediction_pipeline = PredictionPipeline(filename)
        result = prediction_pipeline.predict()

        return jsonify(result)

    except Exception as e:
        print(f"❌ Error during prediction: {e}", flush=True)
        return render_template("index.html", prediction="Error occurred")

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 10000))  # default to 10000 if not provided
    app.run(host="0.0.0.0", port=PORT)
