import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model_path = os.path.join("model", "model.h5")
        self.model = load_model(self.model_path)

    def predict(self):
        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Important normalization step

        preds = self.model.predict(img_array)
        print("Prediction probabilities:", preds)

        result = np.argmax(preds, axis=1)[0]
        print("Predicted class index:", result)

        if result == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{ "image": prediction }]
