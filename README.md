# Sales-Prediction-via-Tree-Models Overview

• Implemented and evaluated three classical boosting and bagging tree models (XGBoost, GBDT, Random Forest) on Rossmann store dataset, and conducted grid search to optimize the hyper-parameters, such as the number of trees and the depth of the tree.

• Conducted feature selection using Pearson correlation, analyzed the correlations between independent variable and dependent variable using visualization plots packages.

• Extracted timestamp features which helped reduce RMSE by 6%, and observed that XGBoost achieved 17% RMSE reduction compared with other tree models.

# Background
Rossmann, founded in 1972, is the largest grocery store in Germany with more than 3000 stores in 7 European countries. From time to time, the stores organize short-term promotions as well as continuous promotions to increase sales. In addition, store sales are influenced by many factors, including promotions, competition, school and national holidays, seasonality and cyclicality.
Data presentation
The data is based on 1115 Rossmann chain stores, with a total of 1017209 sales data (27 characteristics) recorded from January 1, 2013 to July 2015.
The data set covers a total of four files.
- train.csv: historical data with sales volume
- test.csv: historical data without sales
- sample_submission.csv: a sample file submitted in the correct format
- store.csv: some additional information about each store
The data in train.csv contains a total of 9 columns of information.
- store: is the id number of the corresponding store
- DayOfWeek: represents the number of days per week that the store is open
- Data: is the date when the corresponding sales were generated
- Sales: is the historical data of sales
- Customers: is the number of customers who came into the store
- Open: indicates whether the store is open or not
- Promo: indicates whether the store has a promotion on that day
- StateHoliday: and SchoolHoliday indicate whether it is a national holiday or a school holiday, respectively.

## Step 1: Loading data
The Rossmann scenario modeling data contains many information dimensions, such as the number of customers, holidays, and so on. It can also be judged as a typical regression-type modeling problem in supervised learning based on its task objective. First do subsequent analysis of the loaded data before mining the modeling.
The DataFrame.info() operation allows you to view the basic information of the DataFrame data (value distribution, missing value situation, etc.)
## Step 2: EDA exploratory data analysis
The scale of the data involved in this case is relatively large, and cannot directly view the data characteristics by naked eyes, but the understanding of the data distribution characteristics can help us achieve better results in the subsequent mining and modeling. Here we will use Pandas, Matplotlib, Seaborn and other tools introduced before to analyze and visualize the data for understanding.
The IDE use for this part is Jupyter Notebook, which is more convenient for interactive plotting to explore data characteristics.
## Step 3: Data pre-processing (missing values)
Some of the processing methods for missing values include:
- Remove fields (remove columns containing missing values).
- Fill in the missing values (fill in the mean, median, or fit fill, etc.).
- Marking missing values by marking them as special values (e.g. -999) or adding a new column to mark whether a field is missing.
## Step 4: Feature Engineering
- Time features, extracting information such as year, month, day of the week
- Character features are converted to numbers
## Step 5: Benchmark model and evaluation
Define the evaluation criterion function
Since continuous values need to be predicted, a regression model needs to be used. Since this project is a Kaggle competition, the test set is evaluated using Root Mean Square Percentage Error (RMSPE), so only RMSPE can be used here.
### Baseline model evaluation
Construct a regression tree model as the base model for modeling and evaluation. The regression tree we directly use SKLearn's DecisionTreeRegressor, with K-fold cross-validation and grid search for tuning, the main adjustment hyperparameter is the maximum depth max_depth of the tree.
Note that the evaluation criterion here is neg_rmspe, which is the appropriate evaluation criterion for incoming model tuning. GridSearchCV defaults to finding the parameter with the largest scoring_fnc, and directly uses the rmspe metric, the smaller the value, the better the model effect, so it should be taken as negative, thus the larger the value of neg_rmspe, the better the model accuracy.
## Step 6: XGBoost Modeling
Model parameters
XGBoost is a more powerful model with more adjustable parameters, and mainly adjust the following hyperparameters.
- eta: learning rate.
- max_depth: the maximum depth of a single regression tree, smaller leads to underfitting, larger leads to overfitting.
- subsample: between 0 and 1, which controls the proportion of random sampling for each tree. By decreasing the value of this parameter, the algorithm will be more conservative and avoid overfitting. However, if this value is set too small, it may lead to underfitting.
- colsample_bytree: between 0 and 1, used to control the proportion of randomly sampled features per tree.
- num_trees: the number of trees, i.e. the number of iteration steps.
