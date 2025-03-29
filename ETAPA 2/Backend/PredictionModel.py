from joblib import load

class Model:

    def __init__(self,columns):
        self.model = load("assets/modelo.joblib")
    def preprocess(self, data):
        return data
    def make_predictions(self, data):
        data = self.preprocess(data)
        result = self.model.predict(data)
        return result
