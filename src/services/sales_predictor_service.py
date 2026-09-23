import joblib
import pandas as pd

class SalesPredictor:

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, input_data):
        data = pd.DataFrame([input_data])
        prediction = self.model.predict(data)
        return prediction[0]

