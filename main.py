import os
import kaggle
import zipfile
import warnings
import numpy as np 
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import show as plotly_show, write_html
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from decimal import ROUND_HALF_UP, Decimal
import plotly.figure_factory as ff
import matplotlib.colors
import gc

warnings.filterwarnings("ignore")

# Ensure the path for the downloaded files exists
download_path = r'C:\Users\Elish\kaggle'  # Change this to your desired download path
if not os.path.exists(download_path):
    os.makedirs(download_path)

print("Downloading dataset...")
# Download the dataset
kaggle.api.competition_download_files('jpx-tokyo-stock-exchange-prediction', path=download_path)

print("Unzipping dataset...")
# Unzip the downloaded files
zip_path = os.path.join(download_path, 'jpx-tokyo-stock-exchange-prediction.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(download_path)

print("Dataset downloaded and unzipped to", download_path)

print("Loading data...")
# Load the data
train = pd.read_csv(os.path.join(download_path, "train_files/stock_prices.csv"), parse_dates=['Date'])
stock_list = pd.read_csv(os.path.join(download_path, "stock_list.csv"))

# Display basic information about the data
print("The training data begins on {} and ends on {}.\n".format(train.Date.min(), train.Date.max()))
print(train.describe().to_string())

# Exploratory Data Analysis
train_date = train.Date.unique()
returns = train.groupby('Date')['Target'].mean().mul(100).rename('Average Return')
close_avg = train.groupby('Date')['Close'].mean().rename('Closing Price')
vol_avg = train.groupby('Date')['Volume'].mean().rename('Volume')

fig = make_subplots(rows=3, cols=1, shared_xaxes=True)

# Custom colors: Violet, Red, Orange
custom_colors = ['#8A2BE2', '#FF0000', '#FFA500']

for i, (data, color) in enumerate(zip([returns, close_avg, vol_avg], custom_colors)):
    fig.add_trace(go.Scatter(x=train_date, y=data, mode='lines', name=data.name, marker_color=color), row=i+1, col=1)

fig.update_xaxes(rangeslider_visible=False,
                 rangeselector=dict(buttons=list([
                     dict(count=6, label="6m", step="month", stepmode="backward"),
                     dict(count=1, label="1y", step="year", stepmode="backward"),
                     dict(count=2, label="2y", step="year", stepmode="backward"),
                     dict(step="all")])),
                 row=1, col=1)
fig.update_layout(template='plotly_white', title='JPX Market Average Stock Return, Closing Price, and Shares Traded',
                  hovermode='x unified', height=700,
                  yaxis1=dict(title='Stock Return', ticksuffix='%'),
                  yaxis2_title='Closing Price', yaxis3_title='Shares Traded',
                  showlegend=False)

# Show the figure
plotly_show(fig)

# Save the figure as HTML
html_path = os.path.join(download_path, 'jpx_market_analysis.html')
write_html(fig, file=html_path)

print(f"Interactive plot saved to {html_path}")

# Additional Visualization: Yearly Average Stock Returns by Sector
stock_list['SectorName'] = [i.rstrip().lower().capitalize() for i in stock_list['17SectorName']]
stock_list['Name'] = [i.rstrip().lower().capitalize() for i in stock_list['Name']]
train_df = train.merge(stock_list[['SecuritiesCode','Name','SectorName']], on='SecuritiesCode', how='left')
train_df['Year'] = train_df['Date'].dt.year
years = {year: pd.DataFrame() for year in train_df.Year.unique()[::-1]}
for key in years.keys():
    df = train_df[train_df.Year == key]
    years[key] = df.groupby('SectorName')['Target'].mean().mul(100).rename("Avg_return_{}".format(key))
df = pd.concat((years[i].to_frame() for i in years.keys()), axis=1)
df = df.sort_values(by="Avg_return_2021")

fig = make_subplots(rows=1, cols=5, shared_yaxes=True)

# Custom colors for bar chart: Blue for positive returns, Orange for negative returns
positive_color = '#1f77b4'  # Blue
negative_color = '#ff7f0e'  # Orange

for i, col in enumerate(df.columns):
    x = df[col]
    mask = x <= 0
    fig.add_trace(go.Bar(x=x[mask], y=df.index[mask], orientation='h',
                         text=x[mask], texttemplate='%{text:.2f}%', textposition='auto',
                         hovertemplate='Average Return in %{y} Stocks = %{x:.4f}%',
                         marker=dict(color=negative_color, opacity=0.7), name=col[-4:]),
                  row=1, col=i+1)
    fig.add_trace(go.Bar(x=x[~mask], y=df.index[~mask], orientation='h',
                         text=x[~mask], texttemplate='%{text:.2f}%', textposition='auto',
                         hovertemplate='Average Return in %{y} Stocks = %{x:.4f}%',
                         marker=dict(color=positive_color, opacity=0.7), name=col[-4:]),
                  row=1, col=i+1)
    fig.update_xaxes(range=(x.min() - .15, x.max() + .15), title='{} Returns'.format(col[-4:]),
                     showticklabels=False, row=1, col=i+1)
fig.update_layout(template='plotly_white', title='Yearly Average Stock Returns by Sector',
                  hovermode='closest', margin=dict(l=250, r=50),
                  height=600, width=1000, showlegend=False)

# Show the figure
plotly_show(fig)

# Save the additional figure as HTML
additional_html_path = os.path.join(download_path, 'yearly_avg_stock_returns.html')
write_html(fig, file=additional_html_path)

print(f"Additional interactive plot saved to {additional_html_path}")


# Filter the data
train_df = train_df[train_df.Date > '2020-12-23']
print("New Train Shape {}.\nMissing values in Target = {}".format(train_df.shape, train_df['Target'].isna().sum()))

# Distribution of Target
fig = go.Figure()
x_hist = train_df['Target']
fig.add_trace(go.Histogram(x=x_hist * 100,
                           marker=dict(color='#00008B', opacity=0.7,
                                       line=dict(width=1, color='#00008B')),
                           xbins=dict(start=-40, end=40, size=1)))
fig.update_layout(template='plotly_white', title='Target Distribution',
                  xaxis=dict(title='Stock Return', ticksuffix='%'), height=450)
plotly_show(fig)

# Save the Target Distribution figure as HTML
target_distribution_html_path = os.path.join(download_path, 'target_distribution.html')
write_html(fig, file=target_distribution_html_path)
print(f"Target distribution plot saved to {target_distribution_html_path}")

# Distribution by Sector
pal = ['hsl(' + str(h) + ',50%,' + '50%)' for h in np.linspace(0, 360, 18)]
fig = go.Figure()
for i, sector in enumerate(stock_list['SectorName'].unique()[::-1]):
    y_data = train_df[train_df['SectorName'] == sector]['Target']
    fig.add_trace(go.Box(y=y_data * 100, name=sector,
                         marker_color=pal[i], showlegend=False))
fig.update_layout(template='plotly_white', title='Target Distribution by Sector',
                  yaxis=dict(title='Stock Return', ticksuffix='%'),
                  margin=dict(b=150), height=750, width=900)
plotly_show(fig)

# Save the Target Distribution by Sector figure as HTML
target_distribution_by_sector_html_path = os.path.join(download_path, 'target_distribution_by_sector.html')
write_html(fig, file=target_distribution_by_sector_html_path)
print(f"Target distribution by sector plot saved to {target_distribution_by_sector_html_path}")

# Custom colors for sectors in the candlestick chart
sector_colors = px.colors.qualitative.Plotly

# Interactive Candlestick Chart
train_date = train_df.Date.unique()
sectors = train_df.SectorName.unique().tolist()
sectors.insert(0, 'All')
open_avg = train_df.groupby('Date')['Open'].mean()
high_avg = train_df.groupby('Date')['High'].mean()
low_avg = train_df.groupby('Date')['Low'].mean()
close_avg = train_df.groupby('Date')['Close'].mean()
buttons = []

fig = go.Figure()
for i in range(len(sectors)):
    if i != 0:
        open_avg = train_df[train_df.SectorName == sectors[i]].groupby('Date')['Open'].mean()
        high_avg = train_df[train_df.SectorName == sectors[i]].groupby('Date')['High'].mean()
        low_avg = train_df[train_df.SectorName == sectors[i]].groupby('Date')['Low'].mean()
        close_avg = train_df[train_df.SectorName == sectors[i]].groupby('Date')['Close'].mean()

    fig.add_trace(go.Candlestick(x=train_date, open=open_avg, high=high_avg,
                                 low=low_avg, close=close_avg, name=sectors[i],
                                 visible=(True if i == 0 else False),
                                 increasing_line_color=sector_colors[i % len(sector_colors)],
                                 decreasing_line_color=sector_colors[(i + 1) % len(sector_colors)]))

    visibility = [False] * len(sectors)
    visibility[i] = True
    button = dict(label=sectors[i],
                  method="update",
                  args=[{"visible": visibility}])
    buttons.append(button)

fig.update_xaxes(rangeslider_visible=True,
                 rangeselector=dict(
                     buttons=list([
                         dict(count=3, label="3m", step="month", stepmode="backward"),
                         dict(count=6, label="6m", step="month", stepmode="backward"),
                         dict(step="all")]), xanchor='left', yanchor='bottom', y=1.16, x=.01))
fig.update_layout(template='plotly_white', title='Stock Price Movements by Sector',
                  hovermode='x unified', showlegend=False, width=1000,
                  updatemenus=[dict(active=0, type="dropdown",
                                    buttons=buttons, xanchor='left',
                                    yanchor='bottom', y=1.01, x=.01)],
                  yaxis=dict(title='Stock Price'))

# Show the figure
plotly_show(fig)

# Save the candlestick chart as HTML
candlestick_html_path = os.path.join(download_path, 'candlestick_chart.html')
write_html(fig, file=candlestick_html_path)

print(f"Candlestick chart saved to {candlestick_html_path}")

# Visualization of stocks with the highest and lowest returns by sector
stock = train_df.groupby('Name')['Target'].mean().mul(100)
stock_low = stock.nsmallest(7)[::-1].rename("Return")
stock_high = stock.nlargest(7).rename("Return")
stock = pd.concat([stock_high, stock_low], axis=0).reset_index()
stock['Sector'] = 'All'

# Initialize an empty list to collect sector stocks DataFrames
all_sector_stocks = [stock]

for i in train_df.SectorName.unique():
    sector = train_df[train_df.SectorName == i].groupby('Name')['Target'].mean().mul(100)
    stock_low = sector.nsmallest(7)[::-1].rename("Return")
    stock_high = sector.nlargest(7).rename("Return")
    sector_stock = pd.concat([stock_high, stock_low], axis=0).reset_index()
    sector_stock['Sector'] = i
    all_sector_stocks.append(sector_stock)

# Concatenate all sector stocks DataFrames
stock = pd.concat(all_sector_stocks, ignore_index=True)
    
fig = go.Figure()
buttons = []
for i, sector in enumerate(stock.Sector.unique()):
    x = stock[stock.Sector == sector]['Name']
    y = stock[stock.Sector == sector]['Return']
    mask = y > 0
    fig.add_trace(go.Bar(x=x[mask], y=y[mask], text=y[mask], 
                         texttemplate='%{text:.2f}%',
                         textposition='auto',
                         name=sector, visible=(False if i != 0 else True),
                         hovertemplate='%{x} average return: %{y:.3f}%',
                         marker=dict(color='#00008B', opacity=0.7)))  # Dark Blue for positive returns
    fig.add_trace(go.Bar(x=x[~mask], y=y[~mask], text=y[~mask], 
                         texttemplate='%{text:.2f}%',
                         textposition='auto',
                         name=sector, visible=(False if i != 0 else True),
                         hovertemplate='%{x} average return: %{y:.3f}%',
                         marker=dict(color='#8B0000', opacity=0.7)))  # Dark Red for negative returns
    
    visibility = [False] * 2 * len(stock.Sector.unique())
    visibility[i * 2], visibility[i * 2 + 1] = True, True
    button = dict(label=sector,
                  method="update",
                  args=[{"visible": visibility}])
    buttons.append(button)

fig.update_layout(title='Stocks with Highest and Lowest Returns by Sector',
                  template='plotly_white', yaxis=dict(title='Average Return', ticksuffix='%'),
                  updatemenus=[dict(active=0, type="dropdown",
                                    buttons=buttons, xanchor='left',
                                    yanchor='bottom', y=1.01, x=.01)], 
                  margin=dict(b=150), showlegend=False, height=700, width=900)

# Show the figure
plotly_show(fig)

# Save the figure as HTML
stock_returns_html_path = os.path.join(download_path, 'stock_returns_by_sector.html')
write_html(fig, file=stock_returns_html_path)
print(f"Stock returns plot saved to {stock_returns_html_path}")

# Filter for specific stocks
stocks = train_df[train_df.SecuritiesCode.isin([4169, 7089, 4582, 2158, 7036])]
df_pivot = stocks.pivot_table(index='Date', columns='Name', values='Close').reset_index()
pal = ['rgb' + str(i) for i in sns.color_palette("plasma", len(df_pivot.columns))]

fig = ff.create_scatterplotmatrix(df_pivot.iloc[:, 1:], diag='histogram', name='')
fig.update_traces(marker=dict(color=pal, opacity=0.9, line_color='white', line_width=.5))
fig.update_layout(template='plotly_white', title='Scatterplots of Highest Performing Stocks', 
                  height=1000, width=1000, showlegend=False)

# Show the figure
plotly_show(fig)

# Save the scatter plot matrix as HTML
scatter_matrix_html_path = os.path.join(download_path, 'scatter_plot_matrix.html')
write_html(fig, file=scatter_matrix_html_path)
print(f"Scatter plot matrix saved to {scatter_matrix_html_path}")

# Calculate the correlation
corr = train_df.groupby('SecuritiesCode')[['Target', 'Close']].corr().unstack().iloc[:, 1]
stocks = corr.nlargest(10).rename("Return").reset_index()
stocks = stocks.merge(train_df[['Name', 'SecuritiesCode']], on='SecuritiesCode').drop_duplicates()
pal = sns.color_palette("coolwarm", 14).as_hex()
rgb = ['rgba' + str(matplotlib.colors.to_rgba(i, 0.7)) for i in pal]

# Create the bar chart
fig = go.Figure()
fig.add_trace(go.Bar(x=stocks.Name, y=stocks.Return, text=stocks.Return, 
                     texttemplate='%{text:.2f}', name='', width=0.8,
                     textposition='outside', marker=dict(color=rgb, line=dict(color=pal, width=1)),
                     hovertemplate='Correlation of %{x} with target = %{y:.3f}'))
fig.update_layout(template='plotly_white', title='Most Correlated Stocks with Target Variable',
                  yaxis=dict(title='Correlation', showticklabels=False), 
                  xaxis=dict(title='Stock', tickangle=45), margin=dict(b=100),
                  width=800, height=500)

# Show the figure
plotly_show(fig)

# Save the bar chart as HTML
correlation_bar_chart_html_path = os.path.join(download_path, 'correlation_bar_chart.html')
write_html(fig, file=correlation_bar_chart_html_path)
print(f"Correlation bar chart saved to {correlation_bar_chart_html_path}")


# Create the pivot table
df_pivot = train_df.pivot_table(index='Date', columns='SectorName', values='Close').reset_index()
corr = df_pivot.corr().round(2)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
c_mask = np.where(~mask, corr, 100)

# Create the correlation matrix for plotting
c = []
for i in c_mask.tolist()[1:]:
    c.append([x for x in i if x != 100])
    
cor = c[::-1]
x = corr.index.tolist()[:-1]
y = corr.columns.tolist()[1:][::-1]

# Create the annotated heatmap
fig = ff.create_annotated_heatmap(z=cor, x=x, y=y, 
                                  hovertemplate='Correlation between %{x} and %{y} stocks = %{z}',
                                  colorscale='plasma', name='')
fig.update_layout(template='plotly_white', title='Stock Correlation between Sectors',
                  margin=dict(l=250, t=270), height=800, width=900,
                  yaxis=dict(showgrid=False, autorange='reversed'),
                  xaxis=dict(showgrid=False))

# Show the figure
plotly_show(fig)

# Save the heatmap as HTML
heatmap_html_path = os.path.join(download_path, 'sector_correlation_heatmap.html')
write_html(fig, file=heatmap_html_path)
print(f"Correlation heatmap saved to {heatmap_html_path}")

#Feature Engineering

from decimal import ROUND_HALF_UP, Decimal

def adjust_price(price):
    """
    Args:
        price (pd.DataFrame): DataFrame including stock prices
    Returns:
        pd.DataFrame: DataFrame with generated AdjustedClose
    """
    # Transform Date column into datetime
    price.loc[:, "Date"] = pd.to_datetime(price.loc[:, "Date"], format="%Y-%m-%d")

    def generate_adjusted_close(df):
        """
        Args:
            df (pd.DataFrame): stock prices for a single SecuritiesCode
        Returns:
            pd.DataFrame: stock prices with AdjustedClose for a single SecuritiesCode
        """
        # Sort data to generate CumulativeAdjustmentFactor
        df = df.sort_values("Date", ascending=False)
        # Generate CumulativeAdjustmentFactor
        df.loc[:, "CumulativeAdjustmentFactor"] = df["AdjustmentFactor"].cumprod()
        # Generate AdjustedClose
        df.loc[:, "AdjustedClose"] = (
            df["CumulativeAdjustmentFactor"] * df["Close"]
        ).map(lambda x: float(
            Decimal(str(x)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        ))
        # Reverse order
        df = df.sort_values("Date")
        # To fill AdjustedClose, replace 0 into np.nan
        df.loc[df["AdjustedClose"] == 0, "AdjustedClose"] = np.nan
        # Forward fill AdjustedClose
        df.loc[:, "AdjustedClose"] = df.loc[:, "AdjustedClose"].ffill()
        return df
    
    # Generate AdjustedClose
    price = price.sort_values(["SecuritiesCode", "Date"])
    price = price.groupby("SecuritiesCode").apply(generate_adjusted_close).reset_index(drop=True)
    return price

# Drop ExpectedDividend and fill missing values with 0
train = train.drop('ExpectedDividend', axis=1).fillna(0)
# Apply adjust_price function to the training data
prices = adjust_price(train)

def create_features(df):
    df = df.copy()
    col = 'AdjustedClose'
    periods = [5, 10, 20, 30, 50]
    for period in periods:
        df.loc[:, "Return_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].pct_change(period)
        df.loc[:, "MovingAvg_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].rolling(window=period).mean().values
        df.loc[:, "ExpMovingAvg_{}Day".format(period)] = df.groupby("SecuritiesCode")[col].ewm(span=period, adjust=False).mean().values
        df.loc[:, "Volatility_{}Day".format(period)] = np.log(df[col]).groupby(df["SecuritiesCode"]).diff().rolling(period).std()
    return df

# Create features
price_features = create_features(df=prices)
# Drop unnecessary columns
price_features.drop(['RowId', 'SupervisionFlag', 'AdjustmentFactor', 'CumulativeAdjustmentFactor', 'Close'], axis=1, inplace=True)

# Merge and filter the data
price_names = price_features.merge(stock_list[['SecuritiesCode', 'Name', 'SectorName']], on='SecuritiesCode').set_index('Date')
price_names = price_names[price_names.index >= '2020-12-29']
price_names.fillna(0, inplace=True)

# Define features and names for subplots
features = ['MovingAvg', 'ExpMovingAvg', 'Return', 'Volatility']
names = ['Average', 'Exp. Moving Average', 'Period', 'Volatility']
buttons = []

# Create subplots
fig = make_subplots(rows=2, cols=2, 
                    shared_xaxes=True, 
                    vertical_spacing=0.1,
                    subplot_titles=('Adjusted Close Moving Average',
                                    'Exponential Moving Average',
                                    'Stock Return', 'Stock Volatility'))

# Iterate over each sector
for i, sector in enumerate(price_names.SectorName.unique()):
    
    sector_df = price_names[price_names.SectorName == sector]
    periods = [0, 10, 30, 50]
    colors = px.colors.qualitative.D3
    dash = ['solid', 'dash', 'longdash', 'dashdot', 'longdashdot']
    row, col = 1, 1
    
    for j, (feature, name) in enumerate(zip(features, names)):
        if j >= 2:
            row, periods = 2, [10, 30, 50]
            colors = px.colors.qualitative.Bold[1:]
        if j % 2 == 0:
            col = 1
        else:
            col = 2
        
        for k, period in enumerate(periods):
            if (k == 0) & (j < 2):
                plot_data = sector_df.groupby(sector_df.index)['AdjustedClose'].mean().rename('Adjusted Close')
            elif j >= 2:
                plot_data = sector_df.groupby(sector_df.index)['{}_{}Day'.format(feature, period)].mean().mul(100).rename('{}-day {}'.format(period, name))
            else:
                plot_data = sector_df.groupby(sector_df.index)['{}_{}Day'.format(feature, period)].mean().rename('{}-day {}'.format(period, name))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data, mode='lines',
                                     name=plot_data.name, marker_color=colors[k+1],
                                     line=dict(width=2, dash=(dash[k] if j < 2 else 'solid')), 
                                     showlegend=(True if (j == 0) or (j == 2) else False), legendgroup=row,
                                     visible=(False if i != 0 else True)), row=row, col=col)
            
    visibility = [False] * 14 * len(price_names.SectorName.unique())
    for l in range(i * 14, i * 14 + 14):
        visibility[l] = True
    button = dict(label=sector,
                  method="update",
                  args=[{"visible": visibility}])
    buttons.append(button)

# Update layout
fig.update_layout(title='Stock Price Moving Average, Return,<br>and Volatility by Sector',
                  template='plotly_white', yaxis3_ticksuffix='%', yaxis4_ticksuffix='%',
                  legend_title_text='Period', legend_tracegroupgap=250,
                  updatemenus=[dict(active=0, type="dropdown",
                                    buttons=buttons, xanchor='left',
                                    yanchor='bottom', y=1.105, x=.01)], 
                  hovermode='x unified', height=800, width=1200, margin=dict(t=150))

# Show the figure
plotly_show(fig)

# Save the subplot figure as HTML
subplots_html_path = os.path.join(download_path, 'sector_subplots.html')
write_html(fig, file=subplots_html_path)
print(f"Sector subplots saved to {subplots_html_path}")

#Stock Price Prediction

def calc_spread_return_sharpe(df: pd.DataFrame, portfolio_size: int = 200, toprank_weight_ratio: float = 2) -> float:
    """
    Args:
        df (pd.DataFrame): predicted results
        portfolio_size (int): # of equities to buy/sell
        toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
    Returns:
        (float): sharpe ratio
    """
    def _calc_spread_return_per_day(df, portfolio_size, toprank_weight_ratio):
        """
        Args:
            df (pd.DataFrame): predicted results
            portfolio_size (int): # of equities to buy/sell
            toprank_weight_ratio (float): the relative weight of the most highly ranked stock compared to the least.
        Returns:
            (float): spread return
        """
        assert df['Rank'].min() == 0
        assert df['Rank'].max() == len(df['Rank']) - 1
        weights = np.linspace(start=toprank_weight_ratio, stop=1, num=portfolio_size)
        purchase = (df.sort_values(by='Rank')['Target'][:portfolio_size] * weights).sum() / weights.mean()
        short = (df.sort_values(by='Rank', ascending=False)['Target'][:portfolio_size] * weights).sum() / weights.mean()
        return purchase - short

    buf = df.groupby('Date').apply(_calc_spread_return_per_day, portfolio_size, toprank_weight_ratio)
    sharpe_ratio = buf.mean() / buf.std()
    return sharpe_ratio

# Time series cross-validation
ts_fold = TimeSeriesSplit(n_splits=10, gap=10000)
prices = price_features.dropna().sort_values(['Date', 'SecuritiesCode'])
y = prices['Target'].to_numpy()
X = prices.drop(['Target'], axis=1)

feat_importance = pd.DataFrame()
sharpe_ratio = []

for fold, (train_idx, val_idx) in enumerate(ts_fold.split(X, y)):
    print("\n========================== Fold {} ==========================".format(fold + 1))
    X_train, y_train = X.iloc[train_idx, :], y[train_idx]
    X_valid, y_val = X.iloc[val_idx, :], y[val_idx]
    
    print("Train Date range: {} to {}".format(X_train.Date.min(), X_train.Date.max()))
    print("Valid Date range: {} to {}".format(X_valid.Date.min(), X_valid.Date.max()))
    
    X_train.drop(['Date', 'SecuritiesCode'], axis=1, inplace=True)
    X_val = X_valid[X_valid.columns[~X_valid.columns.isin(['Date', 'SecuritiesCode'])]]
    val_dates = X_valid.Date.unique()[1:-1]
    print("\nTrain Shape: {} {}, Valid Shape: {} {}".format(X_train.shape, y_train.shape, X_val.shape, y_val.shape))
    
    params = {
        'n_estimators': 500,
        'num_leaves': 100,
        'learning_rate': 0.1,
        'colsample_bytree': 0.9,
        'subsample': 0.8,
        'reg_alpha': 0.4,
        'metric': 'mae',
        'random_state': 21
    }
    
    gbm = LGBMRegressor(**params).fit(X_train, y_train, 
                                      eval_set=[(X_train, y_train), (X_val, y_val)],
                                      eval_metric=['mae', 'mse'])
    y_pred = gbm.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    feat_importance["Importance_Fold" + str(fold)] = gbm.feature_importances_
    feat_importance.set_index(X_train.columns, inplace=True)
    
    rank = []
    X_val_df = X_valid[X_valid.Date.isin(val_dates)]
    for i in X_val_df.Date.unique():
        temp_df = X_val_df[X_val_df.Date == i].drop(['Date', 'SecuritiesCode'], axis=1)
        temp_df["pred"] = gbm.predict(temp_df)
        temp_df["Rank"] = (temp_df["pred"].rank(method="first", ascending=False) - 1).astype(int)
        rank.append(temp_df["Rank"].values)

    stock_rank = pd.Series([x for y in rank for x in y], name="Rank")
    df = pd.concat([X_val_df.reset_index(drop=True), stock_rank,
                    prices[prices.Date.isin(val_dates)]['Target'].reset_index(drop=True)], axis=1)
    sharpe = calc_spread_return_sharpe(df)
    sharpe_ratio.append(sharpe)
    print("Valid Sharpe: {}, RMSE: {}, MAE: {}".format(sharpe, rmse, mae))
    
    del X_train, y_train, X_val, y_val
    gc.collect()

print("\nAverage cross-validation Sharpe Ratio: {:.4f}, standard deviation = {:.2f}.".format(np.mean(sharpe_ratio), np.std(sharpe_ratio)))

#Feature Importance
# Calculate average feature importance across folds
feat_importance['avg'] = feat_importance.mean(axis=1)
feat_importance = feat_importance.sort_values(by='avg', ascending=True)

# Define color palette
pal = sns.color_palette("viridis", 29).as_hex()[2:]


# Create the plot
fig = go.Figure()

# Add shapes (lines) for each feature
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['avg'][i], 
                       line_color=pal[::-1][i], opacity=0.7, line_width=4))

# Add scatter plot points for each feature
fig.add_trace(go.Scatter(x=feat_importance['avg'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.0f}<extra></extra>'))

# Update layout
fig.update_layout(template='plotly_white', title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance', zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120, t=80),
                  height=700, width=800)

# Show the figure
plotly_show(fig)

# Save the feature importance plot as HTML
feature_importance_html_path = os.path.join(download_path, 'feature_importance.html')
write_html(fig, file=feature_importance_html_path)
print(f"Feature importance plot saved to {feature_importance_html_path}")

# Select top 3 features based on average importance
cols_fin = feat_importance['avg'].nlargest(3).index.tolist()
# Extend the list to include 'Open', 'High', and 'Low' prices
cols_fin.extend(['Open', 'High', 'Low'])

# Prepare the training data
X_train = prices[cols_fin]
y_train = prices['Target']

# Initialize and fit the LightGBM model
gbm = LGBMRegressor(**params).fit(X_train, y_train)

# Print the selected features
print("Selected Features for Final Model Training:", cols_fin)
