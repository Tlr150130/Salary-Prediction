###### libraries ################################################
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from results import save_figure

###### Functions #################################################
### Numerical data
def num_eda_plots(data, col, target_col):
    """
    Creates 3 EDA plots: [lineplot, boxplot, histogram]
    
    Parameters:
    ===========
    data: training dataframe
    col: feature name that we want to plot
    target_col: name of dependent variable
    """
    
    plt.figure(figsize = (12, 4))
    sns.set_style("darkgrid")
    plt.suptitle("Distribution of {}".format(col))
    plt.tight_layout()
    
    ### line plot
    plt.subplot(1, 3, 1)
    # Break num var into chunks to get mean/std of chunk
    y_mean = data.groupby([col]).mean()
    y_mean = y_mean[target_col]
    y_std = data.groupby([col]).std()
    y_std = y_std[target_col]

    # get x variable
    x = y_mean.index
    
    # plot
    plt.plot(x, y_mean, "-b")
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha = 0.3)
    plt.xlabel(col)
    plt.ylabel(target_col)
    
    ### box plot
    plt.subplot(1, 3, 2)    
    sns.boxplot(data = data, x = col)
    
    ### histogram
    plt.subplot(1, 3, 3)
    sns.histplot(data, x = col, bins = 50)
    plt.ylabel(" ")
    
    # save plot
    save_figure("num_eda_plot_{}".format(col))

def num_eda(data, target_col):
    """
    Automatically feeds numerical columns into the plotting function
    
    Parameters:
    ===========
    data: training dataframe
    target_col: name of dependent variable
    """
    
    for col in data.columns[data.dtypes == "int64"].drop(target_col):
        num_eda_plots(data = data, col = col, target_col = target_col)

### Target Feature - Numerical
def target_num_eda(data, target_col):
    """
    Creates 2 EDA plots: [boxplot, histogram]
    
    Parameters:
    ===========
    data: training dataframe
    col: feature name that we want to plot
    target_col: name of dependent variable
    """
    plt.figure(figsize = (10, 4))
    sns.set_style("darkgrid")
    plt.suptitle("Distribution of {}".format(target_col))
    plt.tight_layout()
    
    ### boxplot
    plt.subplot(1, 2, 1)
    sns.boxplot(data = data, x = target_col)
    
    ### histogram
    plt.subplot(1, 2, 2)
    sns.histplot(data = data, x = target_col, bins = 50)
    plt.ylabel(" ")

    # save plot
    save_figure("target_eda_plot_{}".format(target_col))
    
### Categorical Features
def cat_eda_plot(data, col, target_col):
    """
    Creates 2 EDA plots: [histogram, boxplot]
    
    Parameters:
    ===========
    data: training dataframe
    col: feature name that we want to plot
    target_col: name of dependent variable
    """
        
    plt.figure(figsize = (12, 6))
    sns.set(font_scale = 1.2)
    sns.set_style("darkgrid")
    plt.suptitle("Distribution of {}".format(col))   

    ### histogram
    plt.subplot(1, 2, 1)
    counts = np.array(data[col].value_counts().sort_values(ascending = False))
    index = np.array(data[col].value_counts().sort_values(ascending = False).index)
    sns.barplot(x = index, y = counts, color = "salmon") 
    plt.xticks(rotation = 90)
    plt.legend([],[], frameon=False)

    ### boxplot
    plt.subplot(1, 2, 2)    
    col_means = data.groupby(col).mean()
    x_index = col_means[target_col].sort_values(ascending = False).index
    sns.boxplot(data = data, x = col, y = target_col, color = "cornflowerblue" , order = x_index) 
    plt.xticks(rotation = 90)
    plt.legend([],[], frameon=False)
    
    plt.tight_layout()
    
    # save plot
    save_figure("cat_eda_plot_{}".format(col))
    
def cat_eda(data, target_col):
    """
    Automatically feeds categorical columns into the plotting function
    
    Parameters:
    ===========
    data: training dataframe
    target_col: name of dependent variable
    """
    
    for col in data.columns[data.dtypes == "O"].drop(["jobId", "companyId"]):
        cat_eda_plot(data = data, col = col, target_col = target_col)        

### Interactions
def cat_num_interaction_plots(data, cat_var):
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
        plt.tight_layout()
        
        # yearsExperience
        p1 = sns.boxplot(data = data, x = cat_var, y = num_variables[0], ax = axes[0])
        p1.set_xlabel("{}".format(cat_var))
        p1.set_ylabel("{}".format(num_variables[0]))
        axes[0].tick_params(axis='x', rotation=90)
        
        # milesFromMetropolis
        p2 = sns.boxplot(data = data, x = cat_var, y = num_variables[1], ax = axes[1])
        p2.set_xlabel("{}".format(cat_var))
        p2.set_ylabel("{}".format(num_variables[1]))
        axes[1].tick_params(axis='x', rotation=90)
        
        # save plot
        save_figure("cat_num_interaction_plots_{}".format(cat_var))
    
        plt.show()
        
    else:
        print("Please give a categorical and numeric variable")
        
def cat_cat_interaction_plot(data, cat_var_1, cat_var_2):
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
        sns.set(font_scale = 1.2)
        plt.tight_layout()
        sns.catplot(data = variable_averages,
                    x = cat_var_1,
                    y = target_variable,
                    hue = cat_var_2,
                    kind = "point",
                    legend = False)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
        tic = plt.xticks(rotation=90)
        
        # save plot
        save_figure("cat_cat_interaction_plot_{}_{}".format(cat_var_1, cat_var_2))
    
        plt.show()
    # guard statement
    else:
        print("Please enter 2 feature labels.")  

def interaction_plots(data):
    """
    Automatically feeds categorical columns into the plotting functions
    
    Parameters:
    ===========
    data: training dataframe
    """
    # get variables
    cat_variables = data.columns[data.dtypes == "O"].drop(["jobId", "companyId"])
    
    # cat vs. num
    print('\n{0:*^80}\n'.format(' Categorical vs Numerical '))
    for col in cat_variables:
        cat_num_interaction_plots(data = data, cat_var = col)
    
    # cat vs. cat
    print('\n{0:*^80}\n'.format(' Categorical vs Categorical '))   
    for comb in combinations(cat_variables, 2):
        cat_cat_interaction_plot(data = data, cat_var_1 = comb[0], cat_var_2 = comb[1])

### interactions
def corr_heat_map(data, target_col):
    """
    Creates a correlation heat map with all variables except jobId and companyId
    
    Parameters:
    ===========
    data: training dataframe
    """
    # make sure that ID's are dropped
    data = data.drop(["jobId", "companyId"], axis = 1)
    
    # transform categorical levels into numerical using level averages
    cat_variables = data.columns[data.dtypes == "O"]
    for col in cat_variables:
        data[col] = data.groupby(col)[target_col].transform("mean")
        
    # create heatmap
    plt.figure(figsize = (12, 10))
    sns.heatmap(data = data.corr(), cmap = "rocket", annot = True)
    plt.xticks(rotation = 90)
    
    # save plot
    save_figure("corr_heat_map")
 
