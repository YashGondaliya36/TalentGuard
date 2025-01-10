import os
from TalentGuard import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from TalentGuard.config.configuration import DataTransformationConfig
from sklearn.preprocessing import LabelEncoder      



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def encode_data(self,data):
        """Encode the categorical columns in the DataFrame."""

        label_encoder = LabelEncoder()
        data['salary'] = label_encoder.fit_transform(data['salary'])
        data['Department'] = label_encoder.fit_transform(data['Department'])

        # Save the encoded DataFrame as a .pkl file in the artifacts folder
        encoded_data_path = os.path.join(self.config.root_dir, 'encoded_data.pkl')
        data.to_pickle(encoded_data_path)

        logger.info(f"Encoded data saved to {encoded_data_path}")
        return data

    def train_test_spliting(self):
        # Load the data
        data = pd.read_csv(self.config.data_path)

        # Encode the data
        data = self.encode_data(data)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # Save the train and test sets as CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into training and test sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")