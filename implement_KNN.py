import numpy as np 
from collections import Counter

class KNearestNeighbours:
    def __init__(self,k):
        """
        Args:
            Inputs: 
                k (int) : number of nearest neighbours 
        """
        self.k = k 
    
    def fit(self,X:np.array,y:np.array):
        """
        Args:
            Input:
                X(ndarray): Features 
                y(ndarray): Labels
        """
        self.x = X 
        self.y = y
    
    def euclidian_dist(self,ex1:np.array,ex2:np.array):
        """
        Args:
            Input:
                ex1(ndarray) : a row of features
                ex2(ndarray) : a row of features
            Output:
                ndarray: Having eucledian distance between features
         """
        return np.sum((ex1 - ex2)**2) ** 0.5
    
    def get_nearest_neighbours(self,example):
        """
        Args:
            Input:
                example(ndarray) : a row of features
            Output:
                tuple: top k nearest neighbours
         """
        distance  = []
        for i, sample in enumerate(self.x):
            distance.append((i,self.euclidian_dist(sample,example)))
        
        sorted_dist = sorted(distance,key=lambda x:x[1])
        return sorted_dist[:self.k]
    
    def majority_label(self,labels):
        """
        Args:
            Input:
                labels(ndarray) : labels
            Output:
                int: of most voted label
         """
        label_count = Counter(labels)
        highest_count = -1 
        most_voted_label = None
        for label,count in label_count.items():
            if count > highest_count:
                highest_count = count
                most_voted_label = label 
        
        return int(most_voted_label)

    def predict(self,X_test):
        """
        Args:
            Input:
                X_test(ndarray) : Test data
                
            Output:
                list:Of predictions from model
         """
        predictions = []
        for x in X_test:
            idx,_= zip(*self.get_nearest_neighbours(x))
            labels = self.y[np.array(idx)]
            majority = self.majority_label(labels)
            predictions.append(majority)
        
        return predictions

# random data 
X = np.random.rand(200,5)
y = np.random.randint(0,5,200)
X_test = np.random.rand(50,5)
# defining param and model
k = 5
model = KNearestNeighbours(k=5)
model.fit(X,y)
y_pred = model.predict(X_test)
print(y_pred)