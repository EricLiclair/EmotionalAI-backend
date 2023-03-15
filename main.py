import pickle
import pandas as pd
from core.classifier import EmotionClassifier

model_pickle_file_name = "model.sav"
model = EmotionClassifier(model_pickle_file_name=model_pickle_file_name)

X = pd.read_csv("data/test_data.csv")
X = X.drop(columns=X.columns[0], axis=1)
y_result = pd.read_csv("data/test_result.csv")
y = model.predict(X=X)
