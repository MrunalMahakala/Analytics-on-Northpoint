# Analytics-on-Northpoint
North Point makes gaming and educational software.,
To expand their customer base, they teamed up with other companies to share customer lists and send targeted mailings. 
This offers unique opportunities for its members to use its pool of customer lists, which enhances the potential targeted mailing to customers and predictive modeling.



# Problem Statement 
![image](https://github.com/MrunalMahakala/Analytics-on-Northpoint/assets/50626560/7eac05d1-f774-4610-8e5a-f0f65126694e)

  By joining a consortium with a pool of 5,000,000 potential customers. The company is allowed to select 200,000 customers, from which 20,000 were randomly chosen to receive trial mailings. Out of these 20,000 customers, only 1,065 made purchases. To refine their target audience, a subset of 2,000 customers (1,000 purchasers and 1,000 non-purchasers) was created. Given the mailing cost of $2 per customer, reaching all 180,000 potential customers would require an investment of $360,000. To minimize these costs, Northpoint seeks to leverage machine learning models to identify customers likely to purchase and predict their spending, thereby focusing their investment on high-potential customers.

# **Test Mailing and Response Rate**
A test mailing was conducted on 20,000 customers, costing $40,000. The response rate was 5.3%, with 1065 purchases.

# **Data Sampling and Cost Analysis**
The company can mail to 180,000 more customers from a pool of 5 million, at a potential cost of $360,000. A balanced sample of 2000 records (1000 purchasers and 1000 non-purchasers) is used to develop predictive models for targeting potential purchasers and estimating spending.

# Dataset Overview
The dataset contains 2000 observations and 25 columns, including customer demographics, interaction history, and purchase behavior.

# **Exploratory Data Analysis (EDA)**
# Descriptive Statistics
Sequence Number: Ranges from 1 to 2000
Categorical Variables: Sources, web order, gender, address, purchase, US presence (values: 0 or 1)
Numerical Variables: Spending (0 to 1500), interaction days, frequency of purchases
Data Distributions
Numerical Columns: Many customers have low purchase frequencies, high interaction days, and low average spending.
Categorical Columns: Most customers are from the US, few web orders, more male customers, balanced purchasers and non-purchasers.
Sources: Highest records from Sources E and W, highest purchasers from Sources A, U, W, E.
Attribute Analysis
Spending vs Purchase: Most purchases are below $500, average spending around $200-225.
Purchase vs Gender: Slightly more male purchasers.
Source vs Purchase: Sources A, U, W, E have higher purchasers.
Purchase vs US: More purchasers in the US.
Purchase vs Web Order: Higher purchases via web orders.
Freq vs Spend: Majority spend under $500 with low purchase frequency.
Web Order vs Spend: Higher spending via web orders.
Source vs Spend: Highest spending from Source A.
Estimated Gross Profit
Using the response rate (0.053) and average spend ($202.88), the estimated gross profit from mailing 180,000 customers is $1,575,552.

# **Data Preprocessing**
Removed unwanted columns such as sequence numbers.

# **Data Partitioning**
The dataset is split into:

Training set (40%)
Validation set (35%)
Holdout set (25%)
# Feature and Target Selection

**Features:**

US,
Source_*,
Freq,
Last_update_days_ago,
1st_update_days_ago,
Gender (male),
Web order.

**Target Variables:**

Purchase,
Spending.


# **Classification Model Selection**
**Naïve Bayes Model**
Accuracy: 69.57%
Sensitivity: 0.6751

**Logistic Regression Model**
Accuracy: 79%
Sensitivity: 0.79
Used 5-fold cross-validation
**Stepwise Feature Selection**
**Forward Selection**
Sensitivity: 0.80
Accuracy: 78%
**Backward Selection**
Sensitivity:0.79
Accuracy: 79.29%

Forward step-wise logistic regression is chosen for classification due to higher sensitivity.

# **Spending Prediction Model**
**Linear Regression**
Correlation: 0.63
RMSE: 159.28
MAE: 104.88

**Backward Selection**
Correlation: 0.63
RMSE: 160.33
MAE: 105.34

**Forward Selection**
Correlation: 0.63
RMSE: 160.33
MAE: 105.34

**Regression Tree**
Correlation: 0.57
RMSE: 166.47
MAE: 101.48

Linear regression is chosen for predicting spending due to lower RMSE.

# Holdout Set Evaluation
The holdout set is evaluated using the chosen models:

Predictor_purchaser_prob: Probability of purchase from logistic regression

Predicted_spend: Predicted spending from linear regression

Adjusted for oversampling and created columns:
Adjusted_prob_purchaser
Adj_predict_spend

# **Decile and Cumulative Gain Charts**
Decile chart: Shows top 10% of purchasers likely to respond, indicating highest-value customers.
Cumulative gain chart: Demonstrates predictive performance, with the top 10% predicted to generate 3.1 times more profit.

# **Recommendation**
Focus on the top 10% of purchasers to optimize profit margins with minimal investment. Gradually target the next 25% to expand the customer portfolio and enhance marketing strategies.



