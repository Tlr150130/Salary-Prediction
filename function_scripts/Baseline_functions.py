###### libraries ################################################
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

###### Classes and Functions #################################################
class avg_per_industry_degree():
    """
    Creates a model based on averages of categorical variable levels
    
    Methods
    =======
    fit: Calculates the avarage of different levels or combination of levels for prediction
    predict: predicts the target variable based on the averages
    """
    def __init__(self, columns):
        self.fitted_columns = columns
        self.fitted = False
    
    def fit(self, X, y = None):
        """
        Parameters:
        ===========
        X: dataframe
        y: dataframe, series, or numpy array
        columns: list of columns to have the averages based on
        """
        X_copy = X.copy()
        
        if (all(x in X.columns[X.dtypes == "O"] for x in self.fitted_columns)):
            X_copy["target"] = y.copy()
            self.level_averages = X_copy.groupby(self.fitted_columns) \
                                   .mean() \
                                   .drop(["yearsExperience", "milesFromMetropolis"], axis = 1)
            self.fitted = True
            return self
        
        else:
            print("Please choose categorical columns that are in the dataset.")
        
    def predict(self, X):
        """
        Parameters:
        ===========
        X: dataframe
        
        Returns:
        ========
        pred: dataframe["jobId", "target_y"]
        """
        if self.fitted:
            pred = pd.merge(X, 
                            self.level_averages, 
                            how = "left", 
                            left_on = self.fitted_columns, 
                            right_on = self.fitted_columns)[["jobId", "target"]]
            return pred
        
        else:
            print("The model needs to be fitted.")
            
    def get_params(self, deep = False):
        """
        Returns average level values
        """
        if self.fitted:
            return self.level_averages
        else:
            print("The model needs to be fitted.")

def cross_val(model, X, y, cv = 3):
    """
    Return MSE for each cross validation
    
    Parameters:
    ===========
    X: dataframe or numpy array consisting of features
    y: dataframe or 1-D array with target values
    
    Returns:
    ========
    mse: list of mse from each cv iteration
    """
    kf = KFold(n_splits = cv, shuffle = True, random_state = 42)
    mse = []
    
    # run cv's
    for train_index, test_index in kf.split(X):
        features_train, features_test = X.iloc[train_index], X.iloc[test_index]
        target_train, target_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(features_train, target_train)
        
        predictions = model.predict(features_test)
        mse.append(mean_squared_error(predictions["target"], target_test))
        
    return mse
