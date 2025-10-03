import json, pickle, time, platform, sklearn

def load_pickle_model(path="artifacts/production_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)