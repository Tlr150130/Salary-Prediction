# libraries
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.model_selection import cross_val_score
from statistics import mean, stdev

# class
class change_variables(BaseEstimator, TransformerMixin):
    """
    Prepares the data for subsequent modeling
    
    Parameters
    ==========
    cols_to_delete: list(String) of columns to delete
    change_degree: turn degree into a boolean that represents having an advanced degree        
    
    Methods
    =======
    fit: Nothing
    transform: Changes the features of the dataset
    """
    def __init__(self, cols_to_delete = None, change_degree = False, interactions = False):
        self.change_degree = change_degree
        self.interactions = interactions
        self.cols_to_delete = cols_to_delete
            
        
    def fit(self, X, y = None):
        # there is nothing to be fitted
        return self
    
    def transform(self, X, y = None):
        # create a copy
        X_copy = X.copy()
        
        # degree change
        if self.change_degree: 
            X_copy["higher_ed"] = np.where((X['degree'] == "NONE") | (X['degree'] == "HIGH_SCHOOL"), 0, 1)
            X_copy = X_copy.drop("degree", axis = 1)
        
        # interactions between major and industry
        if self.interactions:
            X_copy["major_industry"] = X_copy["major"] + "_" + X_copy["industry"]
        
        # return dataframe result
        if self.cols_to_delete == None:
            return X_copy
        return X_copy.drop(self.cols_to_delete, axis = 1)
    
class to_dense_mat(BaseEstimator, TransformerMixin):
    """
    class to return a desnse matrix for subsequent PCA
    """
    def fit(self, X, y = None):
        return self
   
    def transform(self, X, y = None):
        return X.todense()   

def cv_mse_stats(name, model, X, y):
    """
    Calculates mean MSE, std of MSE
    
    Parameters:
    ===========
    name: String name of the trial
    model: Sklearn algorithm 
    X: Dataframe or Array of features
    y: Dataframe or Array of target values
    
    Returns:
    ========
    mse: array[name, mean(MSE), std(MSE)]
    """
    lin_neg_mse = cross_val_score(model,
                                  X, 
                                  y, 
                                  scoring = "neg_mean_squared_error", 
                                  cv = 5,
                                  verbose = 0,
                                  n_jobs = -1)
    
    lin_mse = -1*lin_neg_mse
    
    mse = [name, mean(lin_mse), stdev(lin_mse)]
    return mse
    