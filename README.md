# Cryptocurrency-trend-prediction
This project was created for my bachelor thesis: "Short term cryptocurrency trend prediction".

I used technical indicators, google trends, blockchain statistics and market metrics as input data for machine learning algorithms:
LGBM Classifier, Logistic Regression, Support Vector Machines, Random Forest Classifier and Long Short Term Memory.

The idea behind this project was to predict next day's price movement for Bitcoin and Ethereum using today's and historical data.

Using Blocked Time Window method I divided data into three subsets where each included a training, validation and test set.

I investigated whether feature selection with RFECV may influence the results. 

Metrics used to evaluate models were: Annualized Return Compounded, Annualized Standard Deviation, Maximum drawdown and Information Ratio.
