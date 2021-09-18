# Libraries
import pandas as pd

"""
Here are the helper functions for data import
"""

# functions
def read_data(dset, return_data = False, verbose = False):
    """
    Parameters
    ----------
    dset: String of either {"train", "test"}
        Denotes which dataset is needed
        
    return_data: Boolean
        Returns the retrieved data
        
    verbose: Boolean
        Denotes if basic dataset characteristics are needed
        

    Returns
    -------
    features: DataFrame 
        returns data if return_data = True

    target_variable: DataFrame 
        returns data if return_data = True
    """   
    
    # get data
    features = pd.read_csv("data/{}_features.csv".format(dset))
    if dset == "train":
        target_variable = pd.read_csv("data/{}_salaries.csv".format(dset))
    
    # basic dataframe characteristics
    if verbose:
        name = ["features", "target variable"]
        i = 0
        for data in [features, target_variable]:
            print('\n{0:*^80}'.format(' Reading in the {} dataset '.format(name[i])))
            print("\nit has {0} rows and {1} columns".format(*data.shape))
            print('\n{0:*^80}\n'.format(' It has the following columns '))
            print(data.dtypes)
            print('\n{0:*^80}\n'.format(' The first 5 rows look like this '))
            print(data.head())
            print("\n")
            i = i + 1
            
    # return data
    if return_data:
        if dset == "train":
            return features, target_variable
        if dset == "test":
            return features
        
def merge_data(df1, df2, common_id):
    """
    Parameters
    ----------
    df1: dataframe
    df2: dataframe
    ***Note: df1 and df2 must share a common key/id column (Ex:)
    
    common_id: String
        the name on the common ey/id column

    Returns
    -------
    merged_data: DataFrame 
        returns merged dataset
    """ 
    
    # join data
    try:
        merged_data = df1.merge(df2, how = "left", on = common_id)
        return merged_data
    except:
        print("Error: Wrong common ID column or there is no common ID column.")

def missing_val(data):
    """
    Paramters
    ---------
    data: dataframe
    """
    print(data.isnull().sum(axis = 0))
    
def target_val_positive(series):
    """
    Parameters
    ----------
    series: pandas series
        target column to check if all values > 0
    """
    if (sum(series < 0) == 0):
        print("All values are positive.")
    else:
        print("Not all values are positive.")
        
def target_val_zero(series):
    """
    Parameters
    ----------
    series: pandas series
        target column to check if any values equal 0
    """
    if (sum(series == 0) == 0):
        print("All values are greater than 0.")
    else:
        print("There are {} values equal to 0.".format(sum(series == 0)))
        
def duplicates(data):
    """
    Checks if there are duplicates in the data
    
    Parameters
    ----------
    data: dataframe
    """
    if (data.duplicated().sum() == 0):
        print("There are no duplicates in the data.")
        
    else:
        print("There are duplicates in the dataset.")
        
def clean_detail_report(data, target_variable):
    """
    Prints a report to see if data needs further cleaning
    
    Parameters
    ----------
    data: dataframe
    """
    # variables
    target_col = target_variable
    
    # title
    print('\n{0:*^80}'.format(' Data Cleanliness Report '))
    # Missing Values
    print("\nMissing Values:")
    missing_val(data)
    # All target variable > 0
    print("\nTarget Variable: {}".format(target_col))
    target_val_positive(data[target_col])
    target_val_zero(data[target_col])
    # Duplicates
    print("\nDuplicates:")
    duplicates(data)
   
def get_data(dset, key = None, target_variable = None, clean_details = False, remove_zeros = False):
    """
    Parameters
    ----------
    dset: String of either {"train", "test"}
        Denotes which dataset is needed
    key: String
        Common column that joins the features and target datasets
    target_variable: String
        Name of the target Variable

    Returns
    -------
    data: DataFrame 
        returns full dataset
    """ 
    # variables
    common_id = key # all datasets should be linked with this id column
    dataset_name = ["train", "test"]
    
    if dset not in dataset_name:
        raise ValueError("Please specify which dataset if needed: {'train', 'test'}")
    
    if dset == "train":
        # get train data
        features, target = read_data(dset, return_data = True)

        # join data
        merged_data = merge_data(features, target, common_id)
    
    if dset == "test":
        # get train data
        test_features = read_data(dset, return_data = True)
        return test_features
        
    # clean data info
    if clean_details:
        if target_variable == None:
            raise ValueError("Please specify the target variable.")
        
        if target_variable in merged_data.columns:
            clean_detail_report(merged_data, target_variable)
        else:
            print("Unable to produce report because specified target variable is not among dataset columns.")
    
    # remove zeros from salary
    if remove_zeros:
        if dset == "train":
            merged_data_no_zeros = merged_data[merged_data["salary"] != 0]
            return merged_data_no_zeros
    
    return merged_data