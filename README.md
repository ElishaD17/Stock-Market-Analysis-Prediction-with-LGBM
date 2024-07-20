# Stock Exchange Prediction Analysis

## Overview
This repository contains scripts and analyses for predicting stock market trends using the JPX Tokyo Stock Exchange dataset from Kaggle. The aim is to leverage financial data to forecast future stock movements and analyze stock performance by sector. This project utilizes machine learning to provide insights into stock returns, volatility, and correlation across different sectors, enhancing stock selection strategies for portfolio optimization.

## Problem Statement
The stock market is inherently volatile and complex, making it challenging for investors to achieve consistent returns. Understanding and predicting stock movements is crucial for effective portfolio management. This project addresses these challenges by analyzing historical stock data to predict future trends and provide actionable insights that can help investors optimize their investment strategies.

## Solution Approach
The solution involves the following steps:
1. **Data Extraction and Preprocessing**: Data is downloaded from the Kaggle JPX Tokyo Stock Exchange competition and processed for analysis.
2. **Exploratory Data Analysis (EDA)**: Initial analysis is performed to understand the trends and distributions of the stock data.
3. **Feature Engineering**: New features are created to capture trends, volatility, and other aspects of the stocks that might affect their future prices.
4. **Model Building**: Machine learning models are employed to predict stock returns. A LightGBM regressor is used due to its effectiveness in handling tabular data.
5. **Performance Evaluation**: The models are evaluated using cross-validation techniques specifically suited for time series data.
6. **Visualization**: Various visualizations are provided to interpret the data and model results effectively, aiding in decision-making.

## Tools Used
- **Python**: Primary programming language used for data manipulation and analysis.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **Plotly & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning model development and evaluation.
- **LightGBM**: For implementing the gradient boosting framework.
- **Kaggle API**: For downloading the dataset directly from Kaggle.
- **Visual Studio Code**: For interactive code execution and result presentation.

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
