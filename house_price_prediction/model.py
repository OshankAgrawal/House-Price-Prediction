from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class HousePriceModel:
    print("Class is called")
    def __init__(self):    
        """
            Initialize the House Price Prediction model.
        """
        self.model = LinearRegression()

    def fit(self, x, y):
        """
            Train the model on features x and target y.
        """
        print("Model fit function called")
        self.model.fit(x, y)

    def predict(self, x):
        """
            Predict house prices for given features x.
        """
        print("Model predict function call")
        return self.model.predict(x)
    
    def evaluate(self, x, y):
        """
            Evaluate the model using Mean Squared Error.
        """
        predictions = self.predict(x)
        return mean_squared_error(y, predictions)