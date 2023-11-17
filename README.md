# ProficiencyPulse
A trained model which can predict the employee performance based on factors as inputs. This will be used to hire employees Recommendations to improve the employee performance based on insights from analysis The given Employee dataset consist of 1200 rows. The features present in the data are 28 columns. The shape of the dataset is 1200x28. The 28 features are classified into quantitative and qualitative where 19 features are quantitative (11 columns consists numeric data & 8 columns consists ordinal data) and 8 features are qualitative. EmpNumber consist alphanumerical data (distinct values) which doesn't play a role as a relevant feature for performance rating.

From Correlation we can get the important aspects of the data, Correlation between features and Performance Rating.Correlation is a statistical measure that expresses the extent to which two variables are linearly related.The analysis of the project has gone through the stage of Univariate,Bivariate & Multivariate analysis, correlation analysis and analysis by each department to satisfy the project goal.

The dataset consists of Categorical data and Numerical data. The Target variable consist of ordinal data, so this is a classification problem.The multiple machine learning model used in this project is Support vector classifier, Random forest classifier & Artifical neural network[Multilayer percepton]. from above all models Artifical neural network[Multilayer percepton] predicts higher accuracy 97%.
1. Requirement

The data was given from the IABAC for this project where the collected source is IABAC™. The data is based on INX Future Inc, (referred as INX ). It is one of the leading data analytics and automation solutions provider with over 15 years of global business presence. INX is consistently rated as top 20 best employers past 5 years. The data is not from the real organization. The whole project was done in Jupiter notebook with python platform.

2. Analysis

Data were analyzed by describing the features present in the data. the features play the bigger part in the analysis. The features tell the relation between the dependent and independent variables. Pandas also help to describe the datasets answering following questions early in our project. The data present in the dataset are divided into numerical and categorical data.

Categorical Features EmpNumber Gender EducationBackground MaritalStatus EmpDepartment EmpJobRole BusinessTravelFrequency OverTime Attrition

Numerical Features Age DistanceFromHome EmpHourlyRate NumCompaniesWorked EmpLastSalaryHikePercent TotalWorkExperienceInYears TrainingTimesLastYear ExperienceYearsAtThisCompany ExperienceYearsInCurrentRole YearsSinceLastPromotion YearsWithCurrManager

Ordinal Features EmpEducationLevel EmpEnvironmentSatisfaction EmpJobInvolvement EmpJobLevel EmpJobSatisfaction EmpRelationshipSatisfaction EmpWorkLifeBalance PerformanceRating

3.Univariate, Bivariate & Multivariate Analysis

Library Used: Matplotlib & Seaborn

Plots Used: Histplot, Lineplot, CountPlot, Barplot

Tip: All Observation or insights written below the plots

Univariate Analysis: In univariate analysis we get the unique labels of categorical features, as well as get the range & density of numbers

Bivariate Analysis: In bivariate analysis we check the feature relationship with target veriable.

Multivariate Analysis: In multivariate Analysis check the relationship between two veriable with respect to the target veriable.

CONCLUSION There are some features are positively correlated with performance rating( Target variable) [Emp Environment Satisfaction,Emp Last Salary Hike Percent,Emp Work Life Balance]

4.Explotary Data Analysis

Basic Check & Statistical Measures* Their is no constant column is present in Numerical as well as categoriacl data.

Distribution of Continuous Features:

In general, one of the first few steps in exploring the data would be to have a rough idea of how the features are distributed with one another. To do so, we shall invoke the familiar distplot function from the Seaborn plotting library. The distribution has been done by both numerical features. it will show the overall idea about the density and majority of data present in a different level.

The age distribution is starting from 18 to 60 where the most of the employees are laying between 30 to 40 age count Employees are worked in the multiple companies up to 8 companies where most of the employees worked up to 2 companies before getting to work here. The hourly rate range is 65 to 95 for majority employees work in this company. In General, Most of Employees work up to 5 years in this company. Most of the employees get 11% to 15% of salary hike in this company.

Check Skewness and Kurtosis of Numerical Features

Checking weather the data is Normally distributed or Not with Skewness and Kurtosis**
YearsSinceLastPromotion, This column is skewed

skewness for YearsSinceLastPromotion: 1.9724620367914252
kurtosis for YearsSinceLastPromotion: 3.5193552691799805
Distribution of Mean of Data
Distribution of mean close to guassian distribution with mean value 9.5
we can say that around 80% feature mean lies between 8.5 to 10.5
Distribution of Standard Deviation of Data
Distribution of standard deviation of data also look like guassian distribution around 30% of feature standard deviation around the range of 3 3 to 20 and remaining 70% feature standard deviation in between 0 to 2
5.Data Pre-Processing

Check Missing Value: Their is no missing value in data

Categorical Data Conversion: Handel categorical data with the help of frequency and mannual encoding, because feature is contain lot's of labels

Mannual Encoding: Mannual encoding is a best techinque to handel categorical feature with the help of map function, map the labels based on frequency.

Frequency Encoding: Frequency encoding is an encoding technique to transform an original categorical variable to a numerical variable by considering the frequency distribution of the data getting value counts.

Outlier Handling Some features are contain outliers so we are impute this outlier with the help of IQR because in all features data is not normally distributed

Feature Transformation: In YearsSinceLastPromotion some skewed & kurtosis is present, so we are use Square Root Transformation techinque

Square root transformation: Square root transformation is one of the many types of standard transformations.This transformation is used for count data (data that follow a Poisson distribution) or small whole numbers. Each data point is replaced by its square root. Negative data is converted to positive by adding a constant, and then transformed.
Q-Q Plot: Q–Q plot is a probability plot, a graphical method for comparing two probability distributions by plotting their quantiles against each other.
Scaling The Data: scaling the data with the help of Standard scalar
Standard Scaling: Standardization is the process of scaling the feature, it assumes the feature follow normal distribution and scale the feature between mean and standard deviation, here mean is 0 and standard deviation is always 1.
6.Future Selection

Drop unique and constant feature: Dropping employee number because this is a constant column as well as drop Years Since Last Promotion because we create a new feaure using square root transformation

Checking Correlation: Checking correlation with the help of heat map, and get the their is no highly correlated feature is present.

Heatmap: A heatmap is a graphical representation of data that uses a system of color-coding to represent different values.

Check Duplicates: In this data Their is no dupicates is present.

PCA: Use pca to reduce the dimension of data, Data is contain total 27 feature after dropping unique and constant column,from PCA it shows the 25 feature has less varaince loss, so we are going to select 25 feature.

Principal component analysis (PCA) is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation, increasing the interpretability of data while preserving the maximum amount of information, and enabling the visualization of multidimensional data. Formally, PCA is a statistical technique for reducing the dimensionality of a dataset.

Saving Pre-Process Data: save the all preprocess data in new file and add target feature to it.
7.Machine learning Model Creation & Evaluation

Define Dependant and Independant Features:

Balancing the data: The data is imbalance, so we need to balance the data with the help of SMOTE

SMOTE: SMOTE (synthetic minority oversampling technique) is one of the most commonly used oversampling methods to solve the imbalance problem. It aims to balance class distribution by randomly increasing minority class examples by replicating them. SMOTE synthesises new minority instances between existing minority instances. 3.Splitting Training And Testing Data: 80% data use for training & 20% data used for testing

Algorithm:

AIM: Create a sweet spot model (Low bias, Low variance)
HERE WE WILL BE EXPERIMENTING WITH THREE ALGORITHM

Support Vector Machine
Random Forest
Artificial Neural Network [MLP Classifier]
Support vector machine well perform on training data with accuracy 96.61% but the test score is 94.66 after applying Hyperparameter tunning score is 98.28 means model is overfit.
Random forest very well perform in training data with 100% accuracy but in testing 95.61% after doing hyperparameter tunning testing score is decreases.
Artifical neural network[Multilayer percepton] perform very well on training data with 98.95% accuracy and testing score is 95.80%. So we are select Artifical neuranl network [Multilayer percepton] model.
