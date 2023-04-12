# Cryptocurrency-trend-prediction
This project was created for my bachelor thesis: "Short term cryptocurrency trend prediction".

I used technical indicators, google trends, blockchain statistics and market metrics as input data for machine learning algorithms:
LGBM Classifier, Logistic Regression, Support Vector Machines, Random Forest Classifier and Long Short Term Memory.

The idea behind this project was to predict next day's price movement for Bitcoin and Ethereum using today's and historical data.

Using Blocked Time Window method I divided data into three subsets where each included a training, validation and test set.

I investigated whether feature selection with RFECV may influence the results. 

Metrics used to evaluate models were: Annualized Return Compounded, Annualized Standard Deviation, Maximum drawdown and Information Ratio.

# Usage 
1. Clone repository to chosen folder
`git clone <url>`
2. In terminal locate Your folder 
'cd path_to_your_folder'
3. Install dependencies
'pip install -r requirements.txt'
4. Open main.py file and run it. You have description of each parameter inside main file, thanks which You are able to choose algorithm or cryptocurrency.
