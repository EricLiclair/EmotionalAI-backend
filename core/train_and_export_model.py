# imports
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from EmotionalAI.settings import BASE_DIR
import os

DEFAULT_MODEL_FILE_LOCATION = os.path.join(BASE_DIR, 'model.sav')

def generate_model(model_pickle_file_name=DEFAULT_MODEL_FILE_LOCATION):
    # dataset
    dataset = os.path.join(BASE_DIR, "data/emotions.csv")
    data = pd.read_csv(dataset)

    # label
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    data['label'] = data['label'].replace(label_mapping)

    # splitting
    X = data.drop('label', axis=1).copy()
    y = data['label'].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=np.random)

    # training
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # export model
    model_pickle_file_name = os.path.join(BASE_DIR, "model.sav")
    with open(model_pickle_file_name, "wb") as model:
        pickle.dump(clf, model)

    print("model saved at location: ", model_pickle_file_name)
