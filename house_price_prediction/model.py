from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class HousePriceModel:
    print("Class is called")
    def __init__(self, test_size=0.2, random_state=42):    
        """
            Initialize the House Price Prediction model.
        """
        self.model = LinearRegression()
        self.test_size = test_size
        self.random_state = random_state

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
            Split data into train and test sets,
            train the model,
            Evaluate the model using Mean Squared Error.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.fit(x_train, y_train)
        predictions = self.predict(x_test)

        return mean_squared_error(y_test, predictions)