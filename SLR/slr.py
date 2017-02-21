
class SimpleLinearRegression():
    """
    Simple implementation of simple linear regression
    """
    
    def __init__(self, train_x, train_y):
        """
        train_x: list
        train_y: list
        """
        self.train_x = train_x
        self.train_y = train_y
        self.mean_x = sum(self.train_x)/len(self.train_x)
        self.mean_y = sum(self.train_y)/len(self.train_y)
        
    def _covariance(self):
        """
        Computes the covariance of train_x and train_y
        return value: float
        """
        mean_train_xy = sum([i*j for i,j in zip(self.train_x,self.train_y)])/len(self.train_x)
        return mean_train_xy - (self.mean_x)*(self.mean_y)
        
    def _variance_x(self):
        """
        Computes the variance of train_x
        return value: float
        """
        mean_squared = self.mean_x**2
        mean_x_squared = sum(i**2 for i in self.train_x)/len(self.train_x)
        return mean_x_squared - mean_squared
        
    def get_parameters(self):
        """
        Computes the parameters for simple linear regression
        return value: list of parameters
        """
        beta_1 = self._covariance()/self._variance_x()
        beta_0 = self.mean_y - (beta_1)*(self.mean_x)
        return [beta_0,beta_1]
        
    def predict(self, test_x):
        """
        Computes the predicted values this model gives
        return values: list
        """
        predicted_values = []
        parameters = self.get_parameters()
        for test in test_x:
            predicted_val = parameters[0] + test*parameters[1]
            predicted_values.append(predicted_val)
        return predicted_values
        
    def _sum_of_errors_squared(self,predicted_values, actual_values):
        """
        Computes the sum of the residuals squared
        predicted_values: list
        actual_values: list
        return value: float
        """
        sse = sum([(i-j)**2 for i,j in zip(actual_values, predict_values)])
        return sse
        
    def score(self,test_x,test_y):
        """
        Computes the training and test errors
        test_x: list
        test_y: list
        return value: string
        """
        predict_values_train = self.predict(self.train_x)
        predict_values_test = self.predict(test_x)
        
        sum_errors_train = self._sum_of_errors_squared(predict_values_train, self.train_y)
        sum_errors_test = self._sum_of_errors_squared(predict_values_test, test_y)
        
        results = "Training error: {}, Test error: {}".format(sum_errors_train,sum_errors_test)
        return results