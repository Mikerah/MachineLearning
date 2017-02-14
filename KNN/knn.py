from math import fabs

class KNearestNeighbors():
    """
    Simple Implementation of the K Nearest Neighbors algorithm
    """

    def __init__(self, train_x, train_y , k=1, type="classification"):
        """
        train_x: list
        train_y: list
        k : integer indicating the number of neareast neighbors
        type: string indicating a classification problem or regression problem
        """
        self.train_x = train_x
        self.train_y = train_y
        self.k = k
        self.type = type
        
        
    def predict(self, test_x):
        """
        Returns the predict for the given test_x values
        test_x: list
        return value: list
        """
        list_of_predicted_values = []
        for i in range(0, len(test_x)):
            # Find the distance between each point in training set and the test point
            list_of_distances = [[self.train_x.index(j),fabs(round(j-test_x[i],2))] for j in self.train_x]
            list_of_distances = sorted(list_of_distances, key=lambda x:x[1])
            
            # Find the neareast data points from the training set closest to the test point
            nearest_neighbors = list_of_distances[:self.k]
            nearest_neighbors_y_vals = []
            
            # Find corresponding y values for the nearest training points
            for k in nearest_neighbors:
                nearest_neighbors_y_vals.append(self.train_y[k[0]])
            
            # We have a classification problem, then find the most occuring classification label, otherwise
            # we have a regression problem and we average the values of the neareast neighbors.
            if self.type == 'classification':
                list_of_predicted_values.append(max(set(nearest_neighbors_y_vals), key=nearest_neighbors_y_vals.count))
            elif self.type == 'regression':
                list_of_predicted_values.append(sum(nearest_neighbors_y_vals)/self.k)
        return list_of_predicted_values
                
        
    def score(self, test_x, test_y):
        """
        Computes the test error and trainin errors
        test_x: list
        test_y: list
        returns value: string
        """
        test_y_predictions = self.predict(test_x)
        train_y_predictions = self.predict(self.train_x)
        correctly_predicted_test = 0
        correctly_predicted_train = 0
        for i in range(len(test_y_predictions)):
            if test_y_predictions[i] == test_y[i]:
                correctly_predicted_test += 1
            if train_y_predictions[i] == self.train_y[i]:
                correctly_predicted_train += 1
        
        average_test = correctly_predicted_test/len(test_y_predictions)
        average_train = correctly_predicted_train/len(train_y_predictions)
        
        results = "Training error: {}, Test error: {}".format(average_train, average_test)
        
        return results