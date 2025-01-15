import joblib
import pandas as pd
from TalentGuard import logger

class EmployeeRetentionRiskPredictor:
    def __init__(self):
        self.model = joblib.load('artifacts/model_trainer/model.joblib')
        self.label_encoders = joblib.load('artifacts/data_transformation/label_encoders.joblib')

    def predict(self, data: pd.DataFrame) -> int:
        logger.info("Predicting Employee leave job or not")
        
        # Create a copy of the input data
        transformed_data = data.copy()
        
        # Transform categorical columns using the saved encoders
        for column, encoder in self.label_encoders.items():
            transformed_data[column] = encoder.transform(transformed_data[column])

        prediction = self.model.predict(transformed_data)
        print(prediction)
        print(transformed_data)

        if prediction == 1:
            msg = "leave job"
        else:
            msg = "Not leave job"

        logger.info(f"Prediction: {msg}")
        return prediction