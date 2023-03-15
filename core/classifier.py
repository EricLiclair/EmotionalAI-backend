import pandas as pd
import pickle
import os
from EmotionalAI.settings import BASE_DIR
from core.train_and_export_model import generate_model
class ClassificationsConstants:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class EmotionClassifier:

    LABEL_TO_INDEX_MAPPING = {
        ClassificationsConstants.NEGATIVE: 0,
        ClassificationsConstants.NEUTRAL: 1,
        ClassificationsConstants.POSITIVE: 2
    }

    INDEX_TO_LABEL_MAPPING = {
        0: ClassificationsConstants.NEGATIVE,
        1: ClassificationsConstants.NEUTRAL,
        2: ClassificationsConstants.POSITIVE
    }

    def __init__(self, model_pickle_file_name: str) -> None:
        """Initializes the class and loads a pickel file as the model"""
        self.loaded_model = self._load_model_from_pickle(
            model_pickle_file_name=model_pickle_file_name)

    def _load_model_from_pickle(self, model_pickle_file_name):
        """Model loading method"""
        if not os.path.exists(model_pickle_file_name):
            print('model not found! Creating...')
            generate_model(model_pickle_file_name=model_pickle_file_name)
        with open(model_pickle_file_name, 'rb') as model_pickle:
            return pickle.load(model_pickle)

    def get_loaded_model(self):
        return self.loaded_model

    def predict(self, X: pd.DataFrame):
        loaded_model = self.get_loaded_model()
        prediction = loaded_model.predict(X)
        return [ self.get_classification_from_label_index(prediction) for prediction in prediction ]

    def get_index_to_label_mapping(self):
        return self.INDEX_TO_LABEL_MAPPING

    def get_classification_from_label_index(self, label_index: int) -> str:
        index_to_label_mapping = self.get_index_to_label_mapping()
        return index_to_label_mapping[label_index]


MODEL_PICKLE_FILE_NAME = os.path.join(BASE_DIR, "model.sav") 

model = EmotionClassifier(model_pickle_file_name=MODEL_PICKLE_FILE_NAME)
