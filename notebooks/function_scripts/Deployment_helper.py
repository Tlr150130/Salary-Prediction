# libraries
import pandas as pd
import os
import joblib
from datetime import datetime

### Functions
def preprocess_pipeline(features): #finish
    """
    Uses a pre-fitted pipeline from 'pipeline.pkl' to process raw data.
    
    Parameters:
    ===========
    features: DataFrame
        raw data
        
    Returns:
    ========
    processed_features: DataFrame
    """
    try:
        PROJECT_ROOT_DIR = "."
        pipeline_path = os.path.join(PROJECT_ROOT_DIR, "model", "pipeline.pkl")
        feature_transformation_pipeline = joblib.load(pipeline_path)
    
    except:
        raise ValueError("There is no 'pipeline.pkl'. Please fit and \
                         save pipeline.")
    
    processed_features = feature_transformation_pipeline.transform(features)
    return processed_features
    
def model_predict(features): #finish
    """
    Uses a pre-trained model from 'best_model.pkl' to predict salary 
    based on inputed features.
    
    Parameters:
    ===========
    features: DataFrame
        processed features
        
    Returns:
    ========
    predictions: Array, shape(n_samples,)
        predicted salary
    """
    try:
        PROJECT_ROOT_DIR = "."
        model_path = os.path.join(PROJECT_ROOT_DIR, "model", "best_model.pkl")
        model = joblib.load(model_path)
    
    except:
        raise ValueError("There is no 'best_model.pkl'. Please fit and \
                         save model.")
        
    predictions = model.predict(features)
    return predictions

def save_predictions(data):
    """
    Saves predictions to a csv file
    
    Parameters:
    ===========
    data: DataFrame
        holds prediction data
    """
    # get date and time for accurate labeling
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d--%H-%M-%S")
    
    # create file path
    PROJECT_ROOT_DIR = "."
    path = os.path.join(PROJECT_ROOT_DIR, 
                        "results", 
                        "Salary_Predictions_{}.csv".format(date_string))
    
    # save file
    data.to_csv(path)
    print("Saved to {}".format(path))
    

def deployment_pipeline(data):
    """
    Pipeline that takes raw data, processes it, and generates predictions
    using the previously found best model. The best model can be updated
    by replacing the best model file.
    
    Parameters: 
    =========== 
    data: DataFrame
        raw data
        
    Returns:
    ========
    results: DataFrame [jobId, predicted_salary]
    """
    # copy features to preserve data integrety
    features = data.copy()
    
    # save Id
    jobId = features["jobId"]
    
    # drop unnecessary features
    selected_features = features.drop(["jobId", "companyId"], axis = 1)
    
    # get pretrained pipeline
    processed_features = preprocess_pipeline(selected_features)
    
    # get predictions
    predictions = model_predict(processed_features)

    # combine jobId with predicted salary    
    predictions_with_id = pd.DataFrame({"jobId": jobId, 
                                        "predicted_salary": predictions})
    
    # save predictions as a csv file
    save_predictions(predictions_with_id) 
    
    # return predictions
    return predictions_with_id