import os
from TalentGuard import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from TalentGuard.config.configuration import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.label_encoders = {
            'salary': LabelEncoder(),
            'Department': LabelEncoder()
        }

    def encode_data(self, data: pd.DataFrame):
        """Encode the categorical columns in the DataFrame."""
        
        # Create a copy of the data to avoid modifying the original
        encoded_data = data.copy()
        
        # Encode each categorical column
        for column, encoder in self.label_encoders.items():
            encoded_data[column] = encoder.fit_transform(encoded_data[column])

        # Save the encoders
        encoders_path = os.path.join(self.config.root_dir, 'label_encoders.joblib')
        joblib.dump(self.label_encoders, encoders_path)
        logger.info(f"Label encoders saved to {encoders_path}")

        return encoded_data

    def train_test_spliting(self):
        # Load the data
        data = pd.read_csv(self.config.data_path)

        # Encode the data
        encoded_data = self.encode_data(data)

        # Split the data into training and test sets
        train, test = train_test_split(encoded_data)

        # Save the train and test sets
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")