# Stock Market Analysis & Prediction with LGBM

## Overview
This project aims to analyze and predict stock market trends using data from the JPX Tokyo Stock Exchange. The analysis utilizes various data science techniques including data visualization, exploratory data analysis (EDA), time series analysis, and machine learning with LightGBM (Light Gradient Boosting Machine).

The goal of this project is to predict stock prices using historical data. The LightGBM model is employed due to its efficiency and high performance in handling large datasets.

## Data Description
The dataset includes stock prices from the JPX Tokyo Stock Exchange. The key features include:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Trading volume

## Data Collection
The dataset is downloaded from Kaggle using the Kaggle API and unzipped to a specified directory. The data is loaded into a pandas DataFrame for analysis.

## Feature Engineering
We adjust stock prices to account for any stock splits or reverse splits and create new features such as moving averages, exponential moving averages, returns, and volatility over different periods (5, 10, 20, 30, and 50 days).

## Exploratory Data Analysis (EDA)
Descriptive statistics are calculated for the dataset, and various visualizations are created to understand the data better. These include:

- Target distribution plot
- Target distribution by sector plot
- Candlestick charts
- Stock returns by sector plot
- Scatter plot matrix
- Correlation bar chart
- Sector correlation heatmap
- Sector subplots

## Model Training
A LightGBM model is trained to predict stock returns using 10-fold cross-validation. The model's performance is evaluated using the Sharpe Ratio, RMSE, and MAE.

## Results
The final average Sharpe Ratio across all folds is `0.1264` with a standard deviation of `0.12`, indicating the model's overall performance. The most important features for predicting stock returns are identified and used for the final model training.

## Conclusion
The project successfully predicts stock returns from the Tokyo Stock Exchange using historical data and advanced machine learning techniques. The visualizations and model results provide valuable insights for investors.

## Interactive Visualizations
The interactive visualizations created in this project can be viewed using the following links:
- [General Market Analysis Plot][![Visualization](https://img.shields.io/badge/Interactive-Visualization-blue)](https://<your-username>.github.io/<repository-name>/jpx_market_analysis.html)
- [Yearly Average Stock Returns by Sector](C:\Users\Elish\kaggle\yearly_avg_stock_returns.html)
- [Target Distribution Plot](C:\Users\Elish\kaggle\target_distribution.html)
- [Target Distribution by Sector Plot](C:\Users\Elish\kaggle\target_distribution_by_sector.html)
- [Candlestick Chart](C:\Users\Elish\kaggle\candlestick_chart.html)
- [Stock Returns by Sector Plot](C:\Users\Elish\kaggle\stock_returns_by_sector.html)
- [Scatter Plot Matrix](C:\Users\Elish\kaggle\scatter_plot_matrix.html)
- [Correlation Bar Chart](C:\Users\Elish\kaggle\correlation_bar_chart.html)
- [Sector Correlation Heatmap](C:\Users\Elish\kaggle\sector_correlation_heatmap.html)
- [Sector Subplots](C:\Users\Elish\kaggle\sector_subplots.html)

## Future Work
Future improvements could involve:
- Incorporating more features like economic indicators
- Trying other machine learning models like XGBoost or neural networks
- Real-time prediction system implementation

## References
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [JPX Tokyo Stock Exchange](https://www.jpx.co.jp/)
- [Kaggle](https://www.kaggle.com/)

## License
This project is licensed under the Apache 2.0 License. See the LICENSE file for more details.
