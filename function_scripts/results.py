# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

### Functions
def save_results(results, filename):
    """
    Saves results as a .pkl file
    
    Parameters:
    ===========
    results: Dataframe 
        MSE results
        
    filename: String
        Desired name of the file
    """
     
    # set folder path
    PROJECT_ROOT_DIR = "."
    folder_path = os.path.join(PROJECT_ROOT_DIR, "results")
    
    # create folder if folder does not exist
    os.makedirs(folder_path, exist_ok = True)
    
    
    # set filepath
    filepath = os.path.join(folder_path, filename + '.pkl')
    
    # save file    
    try:
        results.to_pickle(filepath)
        print("Results Saved to {}".format(filepath))
    except TypeError:
        print("Please input results as a DataFrame")
        
def save_model(results, filename):
    """
    Saves results as a .pkl file
    
    Parameters:
    ===========
    results: estimator 
        
    filename: String
        Desired name of the file
    """
     
    # set folder path
    PROJECT_ROOT_DIR = "."
    folder_path = os.path.join(PROJECT_ROOT_DIR, "model")
    
    # create folder if folder does not exist
    os.makedirs(folder_path, exist_ok = True)
    
    # set filepath
    filepath = os.path.join(folder_path, filename + '.pkl')
    
    # save file         
    joblib.dump(results, filepath)
    print("Results Saved to {}".format(filepath))
        
def plot_results(result_data, title):
    """
    plots the MSE results from different models
    
    Parameters:
    ==========
    result_data: DataFrame
        holds all the MSe results from different models
        
    title: String
        Title of the plot
    """
    
    plt.figure(figsize = (8, 6))
    sns.set(style = "darkgrid")
    plt.tight_layout()
    g = sns.barplot(data = result_data, x = "Name", y = "Mean MSE")
    plt.suptitle(title)
    plt.ylim(325, 425)
    plt.xlabel("")
    
    font = {'weight' : 'normal',
            'size'   : 13}
    
    plt.rc('font', **font)
    
    # add values on columns
    for index, row in result_data.iterrows():
        g.text(row.name, 
               row["Mean MSE"] + 1, 
               round(row["Mean MSE"], 2), 
               color='black', 
               ha="center")  
        
    # save results
    save_figure(title)
        
def display_search_results(grid_search):
    """
    Displays searched parameters and associated mse
    
    Parameters:
    ===========
    grid_search: grid search object
    
    Returns:
    ========
    param_results: Dataframe
        grid search results in dataframe format and sorted
    """
    param_results = pd.DataFrame(grid_search.cv_results_["params"])
    param_results["mean_test_score"] = grid_search.cv_results_["mean_test_score"]
    return param_results.sort_values(by = "mean_test_score")

def save_figure(fig_name, tight_layout=True, fig_extension="png"):
    """
    Saves image into the image folder
    
    Parameters:
    ===========
    fig_name: String
        filename for figure
    
    tight_layout: Boolean
        Condition for tight layout format for plotting
        
    fig_extension: String
        file extension for image
        
    resolution: int
        resolution of image
    """
    # create folder if folder is not created
    PROJECT_ROOT_DIR = "."
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
    os.makedirs(IMAGES_PATH, exist_ok=True)

    # create path to save file
    path = os.path.join(IMAGES_PATH, fig_name + "." + fig_extension)
    print("Saving figure", fig_name)
    
    # add tight layout
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension)
    
def plot_feature_importance(importance, names):
    """
    Create arrays from feature importance and feature names
    
    Parameters:
    ===========
    importance: list of floats
        relative importance of each features according to tree model
        
    names: list of strings
        feature names   
    """
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    feature_importance_df = pd.DataFrame(data = {'feature_names': feature_names, 
                                                 'feature_importance': feature_importance})

    #Sort the DataFrame in order decreasing feature importance
    feature_importance_df.sort_values(by = ['feature_importance'], ascending=False, inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,8))
    
    #Plot Searborn bar chart
    sns.barplot(x = feature_importance_df['feature_importance'], 
                y = feature_importance_df['feature_names'])
    
    #Add chart labels
    title = 'XGBOOST FEATURE IMPORTANCE'
    plt.title(title)
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    
    # save image
    save_figure(title)