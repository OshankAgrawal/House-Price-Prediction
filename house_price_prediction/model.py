from sklearn.linear_model import LinearRegression

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