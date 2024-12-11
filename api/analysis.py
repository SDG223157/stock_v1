#!/usr/bin/env python3
import os
import sys
import traceback
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)

def format_regression_equation(coefficients, intercept, max_x):
    terms = []
    if coefficients[2] != 0:
        terms.append(f"{coefficients[2]:.4f}(x/{max_x})²")
    if coefficients[1] != 0:
        sign = "+" if coefficients[1] > 0 else ""
        terms.append(f"{sign}{coefficients[1]:.4f}(x/{max_x})")
    if intercept != 0:
        sign = "+" if intercept > 0 else ""
        terms.append(f"{sign}{intercept:.4f}")
    equation = "ln(y) = " + " ".join(terms)
    return equation

def get_analysis_dates(end_date, lookback_type, lookback_value):
    """
    Calculate start date based on lookback period
    
    Parameters:
    end_date (str): End date in YYYY-MM-DD format
    lookback_type (str): 'quarters' or 'days'
    lookback_value (int): Number of quarters or days to look back
    
    Returns:
    str: Start date in YYYY-MM-DD format
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if lookback_type == 'quarters':
        start_dt = end_dt - relativedelta(months=3*lookback_value)
    else:  # days
        start_dt = end_dt - relativedelta(days=lookback_value)
    return start_dt.strftime("%Y-%m-%d")

def calculate_price_appreciation_pct(current_price, highest_price, lowest_price):
    """Calculate price appreciation percentage relative to range"""
    total_range = highest_price - lowest_price
    if total_range > 0:
        current_from_low = current_price - lowest_price
        return (current_from_low / total_range) * 100
    return 0

def find_crossover_points(dates, series1, series2, prices):
    """Find points where two series cross each other"""
    crossover_points = []
    crossover_values = []
    crossover_directions = []
    crossover_prices = []
    
    s1 = np.array(series1)
    s2 = np.array(series2)
    
    diff = s1 - s2
    for i in range(1, len(diff)):
        if diff[i-1] <= 0 and diff[i] > 0:
            cross_value = (s1[i-1] + s2[i-1]) / 2
            crossover_points.append(dates[i])
            crossover_values.append(cross_value)
            crossover_directions.append('down')
            crossover_prices.append(prices[i])
        elif diff[i-1] >= 0 and diff[i] < 0:
            cross_value = (s1[i-1] + s2[i-1]) / 2
            crossover_points.append(dates[i])
            crossover_values.append(cross_value)
            crossover_directions.append('up')
            crossover_prices.append(prices[i])
    
    return crossover_points, crossover_values, crossover_directions, crossover_prices

def perform_polynomial_regression(data, future_days=180):
    """Perform polynomial regression analysis"""
    # Transform to log scale
    data['Log_Close'] = np.log(data['Close'])
    
    # Prepare data
    scale = 1
    X = (data.index - data.index[0]).days.values.reshape(-1, 1)
    y = data['Log_Close'].values
    X_scaled = X / (np.max(X) * scale)
    
    # Polynomial regression
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X_scaled)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Generate predictions
    X_future = np.arange(len(data) + future_days).reshape(-1, 1)
    X_future_scaled = X_future / np.max(X) * scale
    X_future_poly = poly_features.transform(X_future_scaled)
    y_pred_log = model.predict(X_future_poly)
    
    # Transform predictions back
    y_pred = np.exp(y_pred_log)
    
    # Calculate confidence bands
    residuals = y - model.predict(X_poly)
    std_dev = np.std(residuals)
    y_pred_upper = np.exp(y_pred_log + 2 * std_dev)
    y_pred_lower = np.exp(y_pred_log - 2 * std_dev)
    
    # Calculate metrics
    r2 = r2_score(y, model.predict(X_poly))
    coef = model.coef_
    intercept = model.intercept_
    max_x = np.max(X)
    
    # Format equation
    equation = format_regression_equation(coef, intercept, max_x)
    
    return {
        'predictions': y_pred,
        'upper_band': y_pred_upper,
        'lower_band': y_pred_lower,
        'r2': r2,
        'coefficients': coef,
        'intercept': intercept,
        'std_dev': std_dev,
        'equation': equation,
        'max_x': max_x
    }

# [continuing in next message due to length...]
def create_combined_analysis(ticker_symbol, end_date=None, lookback_days=365,crossover_days=180):
    """
    Perform combined technical analysis including retracement ratios and polynomial regression
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol
    end_date (str): End date in YYYY-MM-DD format
    lookback_days (int): Days of historical data to analyze
    
    Returns:
    tuple: (summary_df, fig, crossover_df, signals_df) - DataFrames and Plotly figure
    """
    # Handle end date
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get historical data
    ticker = yf.Ticker(ticker_symbol)
    start_date = get_analysis_dates(end_date, 'days', lookback_days)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker_symbol} in the specified date range")
    
    # Convert index to datetime without timezone
    df.index = df.index.tz_localize(None)
    
    # Calculate daily metrics
    df['Daily_Return'] = df['Close'].pct_change()
    daily_volatility = df['Daily_Return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # Calculate retracement metrics
    analysis_dates = []
    ratios = []
    prices = []
    highest_prices = []
    lowest_prices = []
    appreciation_pcts = []
    
    for current_date in df.index:
        year_start = current_date - timedelta(days=crossover_days)
        mask = (df.index > year_start) & (df.index <= current_date)
        year_data = df.loc[mask]
        
        if len(year_data) < 20:
            continue
            
        current_price = year_data['Close'].iloc[-1]
        highest_price = year_data['Close'].max()
        lowest_price = year_data['Close'].min()
        
        # Calculate ratio
        total_move = highest_price - lowest_price
        if total_move > 0:
            current_retracement = highest_price - current_price
            ratio = (current_retracement / total_move) * 100
        else:
            ratio = 0
            
        # Calculate appreciation percentage
        appreciation_pct = calculate_price_appreciation_pct(
            current_price, highest_price, lowest_price)
        
        analysis_dates.append(current_date)
        ratios.append(ratio)
        prices.append(current_price)
        highest_prices.append(highest_price)
        lowest_prices.append(lowest_price)
        appreciation_pcts.append(appreciation_pct)
    
    # Perform polynomial regression
    regression_results = perform_polynomial_regression(df, lookback_days)
    
    # Find crossover points
    crossover_dates, crossover_values, crossover_directions, crossover_prices = \
        find_crossover_points(analysis_dates, ratios, appreciation_pcts, prices)
    
    # Calculate returns from crossover signals with holding period
    signal_returns = []
    current_position = None
    entry_price = None
    total_return = 0
    
    if crossover_dates:
        last_price = prices[-1]  # Current price for open positions
        
        for date, value, direction, price in zip(crossover_dates, crossover_values, 
                                               crossover_directions, crossover_prices):
            if direction == 'up' and current_position is None:  # Buy signal
                entry_price = price
                current_position = 'long'
                signal_returns.append({
                    'Entry Date': date,
                    'Entry Price': price,
                    'Signal': 'Buy',
                    'Status': 'Open'
                })
            elif direction == 'down' and current_position == 'long':  # Sell signal
                exit_price = price
                trade_return = ((exit_price / entry_price) - 1) * 100
                total_return = (1 + total_return/100) * (1 + trade_return/100) * 100 - 100
                current_position = None
                
                # Update the previous buy signal status
                signal_returns[-1]['Status'] = 'Closed'
                
                signal_returns.append({
                    'Entry Date': date,
                    'Entry Price': price,
                    'Signal': 'Sell',
                    'Trade Return': trade_return,
                    'Status': 'Closed'
                })
        
        # Handle open position at the end of the period
        if current_position == 'long':
            open_trade_return = ((last_price / entry_price) - 1) * 100
            total_return = (1 + total_return/100) * (1 + open_trade_return/100) * 100 - 100
            
            # Add open trade return to the last buy signal
            if signal_returns and signal_returns[-1]['Signal'] == 'Buy':
                signal_returns[-1]['Trade Return'] = open_trade_return
                signal_returns[-1]['Current Price'] = last_price
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=analysis_dates,
            y=prices,
            name='Price (Log Scale)',
            line=dict(color='green', width=3),
            yaxis='y2',
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Price</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add regression line and bands
    future_dates = pd.date_range(
        start=df.index[0],
        periods=len(regression_results['predictions']),
        freq='D'
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=regression_results['predictions'],
            name='Regression',
            line=dict(color='blue', width=2, dash='dash'),
            yaxis='y2',
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Predicted</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=regression_results['upper_band'],
            name='Upper Band',
            line=dict(color='lightblue', width=1),
            yaxis='y2',
            showlegend=False,
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Upper Band</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=regression_results['lower_band'],
            name='Lower Band',
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.2)',
            line=dict(color='lightblue', width=1),
            yaxis='y2',
            showlegend=False,
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Lower Band</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add retracement ratio line
    fig.add_trace(
        go.Scatter(
            x=analysis_dates,
            y=ratios,
            name='Retracement Ratio',
            line=dict(color='purple', width=1),
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Ratio</b>: %{y:.1f}%<extra></extra>'
        )
    )
    
    # Add price position line
    fig.add_trace(
        go.Scatter(
            x=analysis_dates,
            y=appreciation_pcts,
            name='Price Position',
            line=dict(color='orange', width=2, dash='dot'),
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Position</b>: %{y:.1f}%<extra></extra>'
        )
    )
    
    # Add crossover points
    if crossover_dates:
        for date, value, direction, price in zip(crossover_dates, crossover_values, 
                                               crossover_directions, crossover_prices):
            color = 'green' if direction == 'up' else 'red'
            formatted_date = date.strftime('%Y-%m-%d')
            base_name = 'Bullish Crossover' if direction == 'up' else 'Bearish Crossover'
            detailed_name = f"{base_name} ({formatted_date}, ${price:.2f})"
            
            fig.add_trace(
                go.Scatter(
                    x=[date],
                    y=[value],
                    mode='markers',
                    name=detailed_name,
                    marker=dict(
                        symbol='star',
                        size=12,
                        color=color,
                        line=dict(width=1, color=color)
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                 '<b>Date</b>: %{x}<br>' +
                                 '<b>Value</b>: %{y:.1f}%<br>' +
                                 '<b>Price</b>: $%{customdata:.2f}<extra></extra>',
                    text=[detailed_name],
                    customdata=[price]
                )
            )
    
    # Add horizontal lines at key levels
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Calculate metrics for annotations
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    days = (df.index[-1] - df.index[0]).days
    annual_return = ((end_price / start_price) ** (365 / days) - 1) * 100

    # Calculate height needed for tables
    base_height = 1000
    table_height = 400  # Additional height for tables
    total_height = base_height + table_height

    # Create Analysis Summary table
    analysis_summary = go.Table(
        domain=dict(x=[0, 0.45], y=[0, 0.25]),  # Position in bottom left
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color='lightgrey',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[
                ['Total Days', 'Current Price', 'Annualized Return', 'Daily Volatility', 
                 'Annual Volatility', 'Regression R²'],
                [
                    f"{days:,d}",
                    f"${end_price:.2f}",
                    f"{annual_return:.2f}%",
                    f"{daily_volatility:.3f}",
                    f"{annualized_volatility:.3f}",
                    f"{regression_results['r2']:.4f}"
                ]
            ],
            align='left',
            font=dict(size=11)
        )
    )

    # Create Trading Signal Analysis table
    if len(signal_returns) > 0:
        trades = []
        buy_signal = None
        
        for signal in signal_returns:
            if signal['Signal'] == 'Buy':
                buy_signal = signal
                if signal['Status'] == 'Open' and 'Trade Return' in signal:
                    trades.append({
                        'Entry Date': signal['Entry Date'].strftime('%Y-%m-%d'),
                        'Entry Price': signal['Entry Price'],
                        'Exit Date': 'Open',
                        'Exit Price': signal['Current Price'],
                        'Return': signal['Trade Return'],
                        'Status': 'Open'
                    })
            elif signal['Signal'] == 'Sell' and buy_signal is not None:
                trades.append({
                    'Entry Date': buy_signal['Entry Date'].strftime('%Y-%m-%d'),
                    'Entry Price': buy_signal['Entry Price'],
                    'Exit Date': signal['Entry Date'].strftime('%Y-%m-%d'),
                    'Exit Price': signal['Entry Price'],
                    'Return': signal['Trade Return'],
                    'Status': 'Closed'
                })
                buy_signal = None

        trading_table = go.Table(
            domain=dict(x=[0.55, 1], y=[0, 0.25]),  # Position in bottom right
            header=dict(
                values=['<b>Entry Date</b>', '<b>Entry Price</b>', '<b>Exit Date</b>', 
                       '<b>Exit Price</b>', '<b>Return</b>', '<b>Status</b>'],
                fill_color='lightgrey',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[
                    [t['Entry Date'] for t in trades],
                    [f"${t['Entry Price']:.2f}" for t in trades],
                    [t['Exit Date'] for t in trades],
                    [f"${t['Exit Price']:.2f}" for t in trades],
                    [f"{t['Return']:.2f}%" for t in trades],
                    [t['Status'] for t in trades]
                ],
                align='left',
                font=dict(size=11)
            )
        )
    else:
        trading_table = go.Table(
            domain=dict(x=[0.55, 1], y=[0, 0.25]),
            header=dict(
                values=['<b>Notice</b>'],
                fill_color='lightgrey',
                align='left'
            ),
            cells=dict(
                values=[['No trading signals found in the analysis period']],
                align='left'
            )
        )

    # Update figure with tables
    fig.add_trace(analysis_summary)
    fig.add_trace(trading_table)
    
    # Update layout
    # Update layout with optimized spacing
    fig.update_layout(
        title=dict(
            text=f'{ticker_symbol} Technical Analysis ({lookback_days} Days)',
            x=0.5,
            xanchor='center',
            font=dict(size=24)
        ),
        height=total_height,
        grid=dict(
            rows=2,
            columns=2,
            pattern='independent'
        ),
        showlegend=True,
        hovermode='x unified',
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref='paper',
                yref='paper',
                text=f'<b>Annualized Return: {annual_return:.2f}%</b>',
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                align='left'
            ),
            dict(
                x=0.45,
                y=0.98,
                xref='paper',
                yref='paper',
                text=(f'<b>Regression Analysis<br>'
                      f'{regression_results["equation"]}<br>'
                      f'R² = {regression_results["r2"]:.4f}</b>'),
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                align='left'
            ),
            dict(
                x=0.15,
                y=0.98,
                xref='paper',
                yref='paper',
                text=(f'<b>Volatility<br>'
                      f'Daily: {daily_volatility:.3f}<br>'
                      f'Annual: {annualized_volatility:.3f}</b>'),
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                align='left'
            ),
            dict(
                x=0.85,
                y=0.98,
                xref='paper',
                yref='paper',
                text=(f'<b>Signal Returns<br>'
                      f'Total Return: {total_return:.2f}%<br>'
                      f'Number of Trades: {len([s for s in signal_returns if s["Signal"] == "Buy"])}</b>'),
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                align='left'
            ),
            # Add table titles
            dict(
                x=0.02,
                y=0.25,
                xref='paper',
                yref='paper',
                text='<b>Analysis Summary</b>',
                showarrow=False,
                font=dict(size=14),
                align='left'
            ),
            dict(
                x=0.55,
                y=0.25,
                xref='paper',
                yref='paper',
                text='<b>Trading Signal Analysis</b>',
                showarrow=False,
                font=dict(size=14),
                align='left'
            )
        ],
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showspikes=True,
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1,
            domain=[0, 0.95]  # Extend x-axis to use more space
        ),
        yaxis=dict(
            title="Ratio and Position (%)",
            ticksuffix="%",
            range=[-10, 120],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showspikes=True,
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1
        ),
        yaxis2=dict(
            title="Price (Log Scale)",
            overlaying="y",
            side="right",
            type="log",
            showgrid=False,
            showspikes=True,
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.12,  # Move legend closer to chart
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(r=50)  # Reduce right margin
    )
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Date': analysis_dates,
        'Price': prices,
        'High': highest_prices,
        'Low': lowest_prices,
        'Retracement_Ratio_Pct': ratios,
        'Price_Position_Pct': appreciation_pcts
    })
    
    # Create crossover dataframe
    crossover_df = pd.DataFrame({
        'Date': crossover_dates,
        'Crossover_Value': crossover_values,
        'Direction': crossover_directions,
        'Price': crossover_prices
    })
    
    # Create signals dataframe
    signals_df = pd.DataFrame(signal_returns)
    
    return summary_df, fig, crossover_df, signals_df
    """
    Perform combined technical analysis including retracement ratios and polynomial regression
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol
    end_date (str): End date in YYYY-MM-DD format
    lookback_days (int): Days of historical data to analyze
    
    Returns:
    tuple: (summary_df, fig, crossover_df, signals_df) - DataFrames and Plotly figure
    """
    # Handle end date
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get historical data
    ticker = yf.Ticker(ticker_symbol)
    start_date = get_analysis_dates(end_date, 'days', lookback_days)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker_symbol} in the specified date range")
    
    # Convert index to datetime without timezone
    df.index = df.index.tz_localize(None)
    
    # Calculate daily metrics
    df['Daily_Return'] = df['Close'].pct_change()
    daily_volatility = df['Daily_Return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # Calculate retracement metrics
    analysis_dates = []
    ratios = []
    prices = []
    highest_prices = []
    lowest_prices = []
    appreciation_pcts = []
    
    for current_date in df.index:
        year_start = current_date - timedelta(days=365)
        mask = (df.index > year_start) & (df.index <= current_date)
        year_data = df.loc[mask]
        
        if len(year_data) < 20:
            continue
            
        current_price = year_data['Close'].iloc[-1]
        highest_price = year_data['Close'].max()
        lowest_price = year_data['Close'].min()
        
        # Calculate ratio
        total_move = highest_price - lowest_price
        if total_move > 0:
            current_retracement = highest_price - current_price
            ratio = (current_retracement / total_move) * 100
        else:
            ratio = 0
            
        # Calculate appreciation percentage
        appreciation_pct = calculate_price_appreciation_pct(
            current_price, highest_price, lowest_price)
        
        analysis_dates.append(current_date)
        ratios.append(ratio)
        prices.append(current_price)
        highest_prices.append(highest_price)
        lowest_prices.append(lowest_price)
        appreciation_pcts.append(appreciation_pct)
    
    # Perform polynomial regression
    regression_results = perform_polynomial_regression(df, lookback_days)
    
    # Find crossover points
    crossover_dates, crossover_values, crossover_directions, crossover_prices = \
        find_crossover_points(analysis_dates, ratios, appreciation_pcts, prices)
    
    # Calculate returns from crossover signals with holding period
    signal_returns = []
    current_position = None
    entry_price = None
    total_return = 0
    
    if crossover_dates:
        last_price = prices[-1]  # Current price for open positions
        
        for date, value, direction, price in zip(crossover_dates, crossover_values, 
                                               crossover_directions, crossover_prices):
            if direction == 'up' and current_position is None:  # Buy signal
                entry_price = price
                current_position = 'long'
                signal_returns.append({
                    'Entry Date': date,
                    'Entry Price': price,
                    'Signal': 'Buy',
                    'Status': 'Open'
                })
            elif direction == 'down' and current_position == 'long':  # Sell signal
                exit_price = price
                trade_return = ((exit_price / entry_price) - 1) * 100
                total_return = (1 + total_return/100) * (1 + trade_return/100) * 100 - 100
                current_position = None
                
                # Update the previous buy signal status
                signal_returns[-1]['Status'] = 'Closed'
                
                signal_returns.append({
                    'Entry Date': date,
                    'Entry Price': price,
                    'Signal': 'Sell',
                    'Trade Return': trade_return,
                    'Status': 'Closed'
                })
        
        # Handle open position at the end of the period
        if current_position == 'long':
            open_trade_return = ((last_price / entry_price) - 1) * 100
            total_return = (1 + total_return/100) * (1 + open_trade_return/100) * 100 - 100
            
            # Add open trade return to the last buy signal
            if signal_returns and signal_returns[-1]['Signal'] == 'Buy':
                signal_returns[-1]['Trade Return'] = open_trade_return
                signal_returns[-1]['Current Price'] = last_price
    
    # Create signals dataframe
    signals_df = pd.DataFrame(signal_returns)
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=analysis_dates,
            y=prices,
            name='Price (Log Scale)',
            line=dict(color='green', width=3),
            yaxis='y2',
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Price</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add regression line and bands
    future_dates = pd.date_range(
        start=df.index[0],
        periods=len(regression_results['predictions']),
        freq='D'
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=regression_results['predictions'],
            name='Regression',
            line=dict(color='blue', width=2, dash='dash'),
            yaxis='y2',
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Predicted</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=regression_results['upper_band'],
            name='Upper Band',
            line=dict(color='lightblue', width=1),
            yaxis='y2',
            showlegend=False,
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Upper Band</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=regression_results['lower_band'],
            name='Lower Band',
            fill='tonexty',
            fillcolor='rgba(173, 216, 230, 0.2)',
            line=dict(color='lightblue', width=1),
            yaxis='y2',
            showlegend=False,
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Lower Band</b>: $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add retracement ratio line
    fig.add_trace(
        go.Scatter(
            x=analysis_dates,
            y=ratios,
            name='Retracement Ratio',
            line=dict(color='purple', width=1),
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Ratio</b>: %{y:.1f}%<extra></extra>'
        )
    )
    
    # Add price position line
    fig.add_trace(
        go.Scatter(
            x=analysis_dates,
            y=appreciation_pcts,
            name='Price Position',
            line=dict(color='orange', width=2, dash='dot'),
            hovertemplate='<b>Date</b>: %{x}<br>' +
                         '<b>Position</b>: %{y:.1f}%<extra></extra>'
        )
    )
    
    # Add crossover points
    if crossover_dates:
        for date, value, direction, price in zip(crossover_dates, crossover_values, 
                                               crossover_directions, crossover_prices):
            color = 'green' if direction == 'up' else 'red'
            formatted_date = date.strftime('%Y-%m-%d')
            base_name = 'Bullish Crossover' if direction == 'up' else 'Bearish Crossover'
            detailed_name = f"{base_name} ({formatted_date}, ${price:.2f})"
            
            fig.add_trace(
                go.Scatter(
                    x=[date],
                    y=[value],
                    mode='markers',
                    name=detailed_name,
                    marker=dict(
                        symbol='star',
                        size=12,
                        color=color,
                        line=dict(width=1, color=color)
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                 '<b>Date</b>: %{x}<br>' +
                                 '<b>Value</b>: %{y:.1f}%<br>' +
                                 '<b>Price</b>: $%{customdata:.2f}<extra></extra>',
                    text=[detailed_name],
                    customdata=[price]
                )
            )
    
    # Add horizontal lines at key levels
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Calculate metrics for annotations
    start_price = df['Close'].iloc[0]
    end_price = df['Close'].iloc[-1]
    days = (df.index[-1] - df.index[0]).days
    annual_return = ((end_price / start_price) ** (365 / days) - 1) * 100
    
    if signal_returns:
        first_signal_date = min(date for date, _, _, _ in zip(crossover_dates, crossover_values, 
                                                            crossover_directions, crossover_prices))
        last_signal_date = max(date for date, _, _, _ in zip(crossover_dates, crossover_values, 
                                                            crossover_directions, crossover_prices))
        years = (last_signal_date - first_signal_date).days / 365.25
        if years > 0:
            signal_cagr = ((1 + total_return/100) ** (1/years) - 1) * 100
        else:
            signal_cagr = 0
    else:
        signal_cagr = 0
        #Update layout
    fig.update_layout(
        title=dict(
            text=f'{ticker_symbol} Technical Analysis ({lookback_days} Days)',
            x=0.5,
            xanchor='center',
            font=dict(size=24)
        ),
        hovermode='x unified',
        showlegend=True,
        height=1000,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref='paper',
                yref='paper',
                text=f'<b>Annualized Return: {annual_return:.2f}%</b>',
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                align='left'
            ),
            dict(
                x=0.25,
                y=0.98,
                xref='paper',
                yref='paper',
                text=(f'<b>Regression Analysis<br>'
                      f'{regression_results["equation"]}<br>'
                      f'R² = {regression_results["r2"]:.4f}</b>'),
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                align='left'
            ),
            dict(
                x=0.02,
                y=0.94,
                xref='paper',
                yref='paper',
                text=(f'<b>Volatility<br>'
                      f'Daily: {daily_volatility:.3f}<br>'
                      f'Annual: {annualized_volatility:.3f}</b>'),
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                align='left'
            ),
            dict(
                x=0.85,  # or whatever x position you're using
                y=0.98,
                xref='paper',
                yref='paper',
                text=(f'<b>Signal Returns<br>'
                    f'Total Return: {total_return:.2f}%<br>'
                    f'CAGR: {signal_cagr:.2f}%<br>'
                    f'Number of Trades: {len([s for s in signal_returns if s["Signal"] == "Buy"])}</b>'),
                showarrow=False,
                font=dict(size=14),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1,
                align='left'
            )
        ],
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showspikes=True,
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1
        ),
        yaxis=dict(
            title="Ratio and Position (%)",
            ticksuffix="%",
            range=[-10, 120],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showspikes=True,
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1
        ),
        yaxis2=dict(
            title="Price (Log Scale)",
            overlaying="y",
            side="right",
            type="log",
            showgrid=False,
            showspikes=True,
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.15,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(r=150)
    )
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Date': analysis_dates,
        'Price': prices,
        'High': highest_prices,
        'Low': lowest_prices,
        'Retracement_Ratio_Pct': ratios,
        'Price_Position_Pct': appreciation_pcts
    })
    
    # Create crossover dataframe
    crossover_df = pd.DataFrame({
        'Date': crossover_dates,
        'Crossover_Value': crossover_values,
        'Direction': crossover_directions,
        'Price': crossover_prices
    })
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Date': analysis_dates,
        'Price': prices,
        'High': highest_prices,
        'Low': lowest_prices,
        'Retracement_Ratio_Pct': ratios,
        'Price_Position_Pct': appreciation_pcts
    })
    
    # Create crossover dataframe
    crossover_df = pd.DataFrame({
        'Date': crossover_dates,
        'Crossover_Value': crossover_values,
        'Direction': crossover_directions,
        'Price': crossover_prices
    })
    
    # Create signals dataframe
    signals_df = pd.DataFrame(signal_returns)
    
    return summary_df, fig, crossover_df, signals_df
    """
    Perform combined technical analysis including retracement ratios and polynomial regression
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol
    end_date (str): End date in YYYY-MM-DD format
    lookback_days (int): Days of historical data to analyze
    
    Returns:
    tuple: (summary_df, fig, crossover_df, signals_df) - DataFrames and Plotly figure
    """
    # Handle end date
def print_signal_analysis(signals_df):
    """
    Print detailed analysis of trading signals including open positions
    
    Parameters:
    signals_df (pandas.DataFrame): DataFrame containing signal information
    """
    if signals_df.empty:
        print("No trading signals found in the analysis period.")
        return
        
    print("\nTrading Signal Analysis:")
    print("-" * 50)
    
    # Group signals by pairs (buy and sell)
    trades = []
    buy_signal = None
    
    for _, row in signals_df.iterrows():
        if row['Signal'] == 'Buy':
            buy_signal = row
            if row['Status'] == 'Open' and 'Trade Return' in row:
                trades.append({
                    'Buy Date': row['Entry Date'],
                    'Buy Price': row['Entry Price'],
                    'Sell Date': 'Open',
                    'Sell Price': row['Current Price'],
                    'Return': row['Trade Return'],
                    'Status': 'Open'
                })
        elif row['Signal'] == 'Sell' and buy_signal is not None:
            trades.append({
                'Buy Date': buy_signal['Entry Date'],
                'Buy Price': buy_signal['Entry Price'],
                'Sell Date': row['Entry Date'],
                'Sell Price': row['Entry Price'],
                'Return': row['Trade Return'],
                'Status': 'Closed'
            })
            buy_signal = None
    
    # Print each trade
    for i, trade in enumerate(trades, 1):
        print(f"\nTrade {i}:")
        print(f"Buy:  {trade['Buy Date'].strftime('%Y-%m-%d')} at ${trade['Buy Price']:.2f}")
        if trade['Status'] == 'Open':
            print(f"Current Position: OPEN at ${trade['Sell Price']:.2f}")
        else:
            print(f"Sell: {trade['Sell Date'].strftime('%Y-%m-%d')} at ${trade['Sell Price']:.2f}")
        print(f"Return: {trade['Return']:.2f}%")
        print(f"Status: {trade['Status']}")
    
    # Calculate and print summary statistics
    if trades:
        returns = [trade['Return'] for trade in trades]
        closed_trades = [t for t in trades if t['Status'] == 'Closed']
        open_trades = [t for t in trades if t['Status'] == 'Open']
        
        print("\nSummary Statistics:")
        print("-" * 50)
        print(f"Total Trades: {len(trades)}")
        print(f"Closed Trades: {len(closed_trades)}")
        print(f"Open Trades: {len(open_trades)}")
        print(f"Average Return per Trade: {np.mean(returns):.2f}%")
        print(f"Best Trade: {max(returns):.2f}%")
        print(f"Worst Trade: {min(returns):.2f}%")
        print(f"Win Rate: {len([r for r in returns if r > 0]) / len(returns) * 100:.1f}%")


def main():
    try:
        debug_print("Starting analysis...")
        
        if len(sys.argv) < 2:
            raise ValueError("Ticker symbol is required")
            
        ticker = sys.argv[1]
        end_date = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
        lookback_days = int(sys.argv[3]) if len(sys.argv) > 3 else 365
        crossover_days = int(sys.argv[4]) if len(sys.argv) > 4 else 180
        
        debug_print(f"Processing {ticker} with params: end_date={end_date}, lookback={lookback_days}, crossover={crossover_days}")
        
        summary_df, fig, crossover_df, signals_df = create_combined_analysis(
            ticker,
            end_date=end_date,
            lookback_days=lookback_days,
            crossover_days=crossover_days
        )
        
        debug_print("Analysis completed, generating HTML...")
        html_content = fig.to_html(
            full_html=True, 
            include_plotlyjs=True,
            config={'responsive': True}
        )
        
        sys.stdout.write(html_content)
        sys.stdout.flush()
        return True
        
    except Exception as e:
        debug_print(f"Error occurred: {str(e)}")
        debug_print(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except Exception as e:
        debug_print(f"Fatal error: {str(e)}")
        debug_print(traceback.format_exc())
        sys.exit(1)

def perform_polynomial_regression(data, future_days=180):
    """Perform polynomial regression analysis"""
    # Transform to log scale
    data['Log_Close'] = np.log(data['Close'])
    
    # Prepare data
    scale = 1
    X = (data.index - data.index[0]).days.values.reshape(-1, 1)
    y = data['Log_Close'].values
    X_scaled = X / (np.max(X) * scale)
    
    # Polynomial regression
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X_scaled)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Generate predictions
    X_future = np.arange(len(data) + future_days).reshape(-1, 1)
    X_future_scaled = X_future / np.max(X) * scale
    X_future_poly = poly_features.transform(X_future_scaled)
    y_pred_log = model.predict(X_future_poly)
    
    # Transform predictions back
    y_pred = np.exp(y_pred_log)
    
    # Calculate confidence bands
    residuals = y - model.predict(X_poly)
    std_dev = np.std(residuals)
    y_pred_upper = np.exp(y_pred_log + 2 * std_dev)
    y_pred_lower = np.exp(y_pred_log - 2 * std_dev)
    
    # Calculate metrics
    r2 = r2_score(y, model.predict(X_poly))
    coef = model.coef_
    intercept = model.intercept_
    max_x = np.max(X)
    
    # Format equation
    equation = format_regression_equation(coef, intercept, max_x)
    
    return {
        'predictions': y_pred,
        'upper_band': y_pred_upper,
        'lower_band': y_pred_lower,
        'r2': r2,
        'coefficients': coef,
        'intercept': intercept,
        'std_dev': std_dev,
        'equation': equation,
        'max_x': max_x
    }

