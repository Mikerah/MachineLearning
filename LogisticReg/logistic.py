class LogisticRegression:
    """
    Simple Implementation of Logistic Regression
    """
    
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        
    def get_params(self):
        """
        Returns the parameters for logistic regression
        return value: list of parameters
        """
        pass
        
    def predict(self, test_x):
        """
        Returns the predictions for the test set
        test_x: list
        return value: list
        """
        pass
        
    def probability(self, test_x):
        """
        Returns the probability of getting a 1
        test_x: list
        return value: float
        """
        pass
        
    def score(self, test_x,test_y):
        """
        Returns the training and test errors
        test_x: list
        test_y: list
        return value: string
        """
        pass