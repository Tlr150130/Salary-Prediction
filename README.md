# Salary Prediction
### Understanding the impact job listing characteristics have on salary

----
## The Business Problem
#### Purpose:
> This model can help job searchers determine whether a job listing offers a reasonable salary based on the requirements and distinct characteristics compared to other jobs with similar requirements and characteristics. Additionally, this model offers applicants leverage when negotiating salaries if they decide to apply to listings with seemingly unreasonable salaries.

> We are using the [Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) to calculate the model's accuracy and determine the best model. MSE is chosen over other regression error metrics because it penalizes predictions that are farther away from the target value as opposed to [Mean Absolute Error (MAE)](https://en.wikipedia.org/wiki/Mean_absolute_error).

<p align="center">
  <img src="https://github.com/Tlr150130/Salary-Prediction/blob/main/README_images/salary-negotiation.jpg" width="500">
</p>

----
## Orientation of Important Folders in Repository
```
$ tree 
.
├── README_images
├── data
│   └── data.zip
├── function_scripts
│   ├── Baseline_functions.py
│   ├── Deployment_helper.py
│   ├── EDA_helper_functions.py
│   ├── Preprocessing.py
│   ├── data_import_functions.py
│   └── results.py
├── images
├── model
│   ├── best_model.pkl
│   └── pipeline.pkl
├── notebooks
│   ├── data
│   ├── function_scripts
│   ├── 00-Getting_Started.ipynb
│   ├── 01-Exploratory_Data_analysis.ipynb
│   ├── 02-Baseline_model.ipynb
│   ├── 03-Feature_Engineering.ipynb
│   ├── 04-Train_models.ipynb
│   └── 05-Model_Deployment.ipynb
├── results
│   ├── Model_MSE_results.plk.pkl
│   └── baseline_model_results.pkl
└── README.md
```
> Note: Only the notebooks folder is necessary for running the code through the notebooks. All helper files and data are placed in the notebook folder.
> 
----
## [Data Wrangling](https://nbviewer.jupyter.org/github/Tlr150130/Salary-Prediction/blob/main/00-Getting_Started.ipynb)
#### The historical data is stored as a csv files:
>* **train_salaries:** Each row has an ID and associated salary value.
>* **train_features:** Each row represents metadata for an individual job posting with its associated ID
>* **test_features:** Same format as train_features

#### Target Variable
> The target variable that we will be trying to predict is the **salary** of the job listing.

#### Feature structure
>* **jobId:** Distinct key that identifies each job listing
>* **companyId:** Distinct key that identifies each company
>* **jobType:** Defines the level of the position
>* **degree:** Represents the education level requirement
>* **major:** Identifies the desired major for the job listing
>* **industry:** Characterizes the specific industry of the job listing
>* **yearsExperience:** Designates the required years of experience for the job listing 
>* **milesFromMetropolis:** Specifies the distance the job listing is from the center of the metropolis
----
## [Exploratory Data Analysis](https://nbviewer.jupyter.org/github/Tlr150130/Salary-Prediction/blob/main/notebooks/01-Exploratory_Data_analysis.ipynb)
#### Target Distribution
> The "salary" variable seems to be slightly right tailed. However, the distribution indicated by the histogram seems to approximate a normal distribution. The removal of the potential outliers could affect the model predictive power towards job listings with higher salary ranges. For that reason, we will keep the observations.

<p align="center">
  <img src="https://github.com/Tlr150130/Salary-Prediction/blob/main/README_images/target_eda_plot_salary.png" width="800">
</p>

#### Categorical Variable Distribution
> The "salary" variable seems to be slightly right tailed. However, the distribution indicated by the histogram seems to approximate a normal distribution. The removal of the potential outliers could affect the model predictive power towards job listings with higher salary ranges. For that reason, we will keep the observations.

<p align="center">
  <img src="https://github.com/Tlr150130/Salary-Prediction/blob/main/README_images/target_eda_plot_salary.png" width="800">
</p>

#### Numerical Variable Distribution
> The numerical features that are being analyzed are "yearExperience", and "milesFromMetropolis". From the line graph below, the both numerical variables are correlated with the target variable, but in opposite directions. Furthermore, the distribution seem to be uniform based on the histogram and boxplots

<p align="center">
  <img src="https://github.com/Tlr150130/Salary-Prediction/blob/main/README_images/num_EDA.png" width="800">
</p>

#### Interactions
> Only the AB interaction plot between major and industry suggests that there is visibly significant interaction between the variables. This suggests that including an interaction variable for linear models could improve the error metric. Additionally, a full polynomial transformation of degree 2 will be considered.

<p align="center">
  <img src="https://github.com/Tlr150130/Salary-Prediction/blob/main/README_images/cat_cat_interactions.png" width="800">
  <img src="https://github.com/Tlr150130/Salary-Prediction/blob/main/README_images/cat_num_interactions.png" width="800">
</p>

#### Correlation via heat map
> Each feature is correlated with the target variable and suggests that each feature is influencial in the model. The heat map suggests that there is multicolinearity amongst the major, degree, and potentially jobType.

<p align="center">
  <img src="https://github.com/Tlr150130/Salary-Prediction/blob/main/README_images/corr_heat_map.png" width="800">
</p>

----
## [Establish Baseline Models](https://nbviewer.jupyter.org/github/Tlr150130/Salary-Prediction/blob/main/notebooks/02-Baseline_model.ipynb)
#### Proposed Baseline Models:
>1. Predict the average per industry and degree
>2. Predict the average per job title
>3. Standard linear regression using categorical level averages

#### Best Preforming Baseline
> The simple linear regression vastly outperformed the other naive baseline models. This suggests that a model will improve any average guess by a large margin. In conclusion, the MSE benchmark to surpass is 399.131258.

<p align="center">
  
Estimator	| Average MSE	| Standard Deviation MSE
----------|-------------|------------------------
Simple Linear Regression	| 399.131258	| 2.084684
Job Type Average	| 963.944600	| 3.304819
Industry and Degree Average |	1125.587248	 | 4.999431
  
</p>

----
## [Feature Engineering](https://nbviewer.jupyter.org/github/Tlr150130/Salary-Prediction/blob/main/notebooks/03-Feature_Engineering.ipynb)
#### one-hot encoding
> Categorical features are encoded via one-hot encoding because there are no hierachial levels in the categorical features. The degree feature tested 

* standard Scaling

----
## [Model Development](https://nbviewer.jupyter.org/github/Tlr150130/Salary-Prediction/blob/main/notebooks/04-Train_models.ipynb)
#### Proposed Models:
<u>Linear Model:</u> 
* Linear Regression
* Polynomial Linear Regression 
* Ridge Regression
* Polynomial Ridge Regression 

<u> Tree-based Model: </u>
* Random-Forest Model
* XGBoost Regresssor

* Results
* image of results

----
## [Model Deployment](https://nbviewer.jupyter.org/github/Tlr150130/Salary-Prediction/blob/main/notebooks/05-Model_Deployment.ipynb)
* For the model be useful, the model needs to be immediately ready to be put into production. A function pipeline was created to take raw inputs with the same format as the training data, process the data, and generate predictions using the best model.
* creat table with 

----
## Summary
> A second order polynomial linear regression model produced the most accurate predictions on salary based on the job listing characteristics and requirements. The model obtained a mean squared error (MSE) of 354 which improved the MSE of the baseline linear model by 11.5%. The best model and associated pipeline is stored as a pickle file and is utilized directly by the deployment function for quick implementation into production.







