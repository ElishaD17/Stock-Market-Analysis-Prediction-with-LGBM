# Stock Market Analysis & Prediction with LGBM

## Overview
This project aims to analyze and predict stock market trends using data from the JPX Tokyo Stock Exchange. The analysis utilizes various data science techniques including data visualization, exploratory data analysis (EDA), time series analysis, and machine learning with LightGBM (Light Gradient Boosting Machine).

## Table of Contents
- Introduction
- Data Description
- Exploratory Data Analysis
- Data Preprocessing
- Model Building
- Model Evaluation
- Conclusion
- Future Work
- How to Use
- References

## Introduction
The goal of this project is to predict stock prices using historical data. The LightGBM model is employed due to its efficiency and high performance in handling large datasets.

## Data Description
The dataset includes stock prices from the JPX Tokyo Stock Exchange. The key features include:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Trading volume

## Exploratory Data Analysis
EDA involves visualizing the data to understand its structure and patterns. Key analyses include:
## Interactive Visualizations

You can view the interactive visualizations [here](file:///C:/Users/Elish/Downloads/jpx_market_analysis.html).
- Trend analysis over time
- Seasonal patterns
- Correlation between different stock prices

## Data Preprocessing
Data preprocessing steps include:
- Handling missing values
- Feature engineering (e.g., creating lag features)
- Normalizing or scaling features if necessary

## Model Building
The model is built using the LightGBM algorithm, which is suitable for large datasets and provides robust performance. Key steps include:
- Splitting data into training and validation sets
- Hyperparameter tuning
- Training the model

## Model Evaluation
Model performance is evaluated using metrics such as:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R²)

## Conclusion
The model provides a predictive framework for stock prices, showing significant potential in forecasting trends. 

## Future Work
Future improvements could involve:
- Incorporating more features like economic indicators
- Trying other machine learning models like XGBoost or neural networks
- Real-time prediction system implementation

## How to Use
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/jpx-stock-market-analysis.git
   cd jpx-stock-market-analysis
   ```

2. **Install the necessary libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook JPX_Stock_Market_Analysis.ipynb
   ```

4. **Explore and modify the code** to fit your specific needs.

## References
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [JPX Tokyo Stock Exchange](https://www.jpx.co.jp/)
- [Kaggle](https://www.kaggle.com/)

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.
