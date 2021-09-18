# libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# functions
def cat_var_counts(data):
    """
    Prints the counts for levels of all categorical variables
    
    Parameters
    ----------
    data: DataFrame
        dataframe with all variables to be analyzed    
    """
    # get categorical variables
    cat_var = data.columns[data.dtypes == "O"][1:]
    
    # counts
    for var in cat_var:
        print("\n{}".format(var))
        print(data[var].value_counts())

def plot_num_var(data):
    """
    Plot a histogram of all numerical variables
    
    Parameters
    ----------
    data: dataframe
    """
    # hard code specific target variable
    target = "salary"
    
    # get numerical variables
    num_var = data.columns[data.dtypes == "int64"].drop(target)
    
    # plot
    sns.set_style("darkgrid")
    plt.figure(figsize=(10,5))
    # fig.subtitle("Numberical Varaibles")
    for var, i in zip(num_var, range(1,3)):
        plt.subplot(1, 2, i)
        plt.hist(data[var], bins = 25)
        plt.title('Distribution of {}'.format(var))
    plt.show()
    
def plot_cat_to_target(data, cat_var):
    """
    Plots barplot of target variable with cat variable
    
    Parameters
    ----------
    data: DataFrame 
    cat_var: String
        name of categorical variable to be plotted
    """
    # hard code specific target variable
    target_var = "salary"
    
    # get categorical variables
    cat_variables = data.columns[data.dtypes == "O"][2:]
    
    # guard
    if cat_var in cat_variables:
        # barplots
        plt.figure(figsize= (12, 8))
        sns.set(font_scale = 1.1)
        p = sns.boxplot(data = data, x = cat_var, y = target_var)
        p.set_xlabel("{}".format(cat_var), fontsize = 20)
        p.set_ylabel("{}".format(target_var), fontsize = 20)
        p.axes.set_title("Distributions of {} Based on {}".format(target_var, cat_var), 
                         fontsize = 30)
        plt.show()

    else: 
        print("Please enter a categorical variable other than jobID and compId")
        
def num_var_scatterplot(data, target_val):
    """
    Plot scatterplots against the target variable
    
    Parameters
    ----------
    data: Dataframe
    target_val: String
        name of the target variable
    """
    if target_val not in data.columns:
        print("Please enter the target variable in the data.")
        
    else:
        # get numerical variables
        num_var = data.columns[data.dtypes == "int64"].drop(target_val)

        # plot 
        sns.set_style("darkgrid")
        plt.figure(figsize = (12, 8))
        fig, axs = plt.subplots(ncols = 2, sharex = False)  
        fig.suptitle("Numeric Variable Distributions")
        for var, index in zip(num_var, range(2)):
            if index == 0:
                sns.scatterplot(data = data, 
                                x = var, 
                                y = "salary", 
                                ax = axs[index], 
                                alpha = 0.2)\
                .set(xlabel = var)
            else:
                sns.scatterplot(data = data, 
                                x = var, 
                                y = "salary", 
                                ax = axs[index], 
                                alpha = 0.2)\
                .set(xlabel = var, ylabel = None)
        plt.show()
        
def cat_vs_cat(data, cat_var_1, cat_var_2):
    """
    Plots a lineplot between 2 categorical variables in the data
    
    Parameters
    ==========
    data: dataframe
    cat_var_1: String
        name of the 1st categorical variable
    cat_var_2: String
        name of the 2nd categorical variable
    """
    # set target variable
    target_variable = "salary"
    
    # plot
    if (cat_var_1 in data.columns and cat_var_2 in data.columns):        
        # target variable
        variable_averages = data.groupby([cat_var_1, cat_var_2], as_index = False).mean() \
                                .drop(["yearsExperience", "milesFromMetropolis"], axis = 1)
        
        # plot
        sns.set(style="darkgrid")
        plt.figure(figsize = (12, 8))
        sns.catplot(data = variable_averages,
                    x = cat_var_1,
                    y = target_variable,
                    hue = cat_var_2,
                    kind = "point",
                    legend = False)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
        tic = plt.xticks(rotation=45)
        plt.show()
    
    # guard statement
    else:
        print("Please enter 2 feature labels.")  

def cat_vs_num(data, cat_var):
    """
    Plots 2 boxplots between a categorical variable and the numeric variables
    
    Parameters
    ==========
    data: dataframe
    cat_var: String
        name of the categorical variable
    num_var: String
        name of the numeric variable
    """
    # get variables
    cat_variables = data.columns[data.dtypes == "O"]
    num_variables = data.columns[data.dtypes == "int64"].drop("salary")
    
    # guard condition
    if cat_var in cat_variables:
        # plot formating
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
        fig.suptitle('{}'.format(cat_var))
        sns.set(style="darkgrid")
        sns.set(font_scale = 1.1)
        
        # yearsExperience
        p1 = sns.boxplot(data = data, x = cat_var, y = num_variables[0], ax = axes[0])
        p1.set_xlabel("{}".format(cat_var))
        p1.set_ylabel("{}".format(num_variables[0]))
        axes[0].tick_params(axis='x', rotation=45)
        
        # milesFromMetropolis
        p2 = sns.boxplot(data = data, x = cat_var, y = num_variables[1], ax = axes[1])
        p2.set_xlabel("{}".format(cat_var))
        p2.set_ylabel("{}".format(num_variables[1]))
        axes[1].tick_params(axis='x', rotation=45)
        plt.show()
        
    else:
        print("Please give a categorical and numeric variable")
        
