from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import plotly
import requests
import logging
from datetime import datetime, timedelta
from libs_utils.tv_datafeed import TvDatafeed, Interval

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PnL Dashboard")
templates = Jinja2Templates(directory="templates")

def fetch_hyperliquid_data(user_address: str, days_back: int = 56):
    """Fetch trading data from Hyperliquid API"""
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (days_back * 24 * 60 * 60 * 1000)

    payload = {
        "type": "userFillsByTime",
        "user": user_address,
        "startTime": start_time,
        "endTime": end_time,
        "aggregateByTime": True
    }

    try:
        response = requests.post("https://api.hyperliquid.xyz/info", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Fetched {len(data)} trades")
        return data
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def fetch_btc_data(start_date: datetime, end_date: datetime):
    """Fetch BTC price data for the exact same period as portfolio trades"""
    try:
        tv = TvDatafeed()

        # Calculate the number of hours needed from start to end
        time_diff = end_date - start_date
        hours_needed = int(time_diff.total_seconds() / 3600) + 48  # Add buffer

        # Fetch BTC data from Binance
        btc_data = tv.get_hist(
            symbol="BTCUSDT",
            exchange="BINANCE",
            interval=Interval.in_1_hour,
            n_bars=min(hours_needed, 2000)  # Limit to avoid excessive data
        )

        if btc_data is not None and not btc_data.empty:
            # Filter BTC data to match the exact trading period
            btc_data_filtered = btc_data[
                (btc_data.index >= start_date) &
                (btc_data.index <= end_date)
            ].copy()

            if not btc_data_filtered.empty:
                print(f"✅ Fetched {len(btc_data_filtered)} BTC price points for exact trading period")
                return btc_data_filtered
            else:
                print("❌ No BTC data found for the trading period")
                return None
        else:
            print("❌ No BTC data received")
            return None

    except Exception as e:
        print(f"❌ Error fetching BTC data: {e}")
        return None

def process_data(raw_data):
    """Process raw data into DataFrame"""
    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)

    # Convert columns
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    df['pnl'] = pd.to_numeric(df['closedPnl'], errors='coerce') - pd.to_numeric(df['fee'], errors='coerce')
    df['price'] = pd.to_numeric(df['px'], errors='coerce')
    df['size'] = pd.to_numeric(df['sz'], errors='coerce')

    # Sort and add trade numbers
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['trade_number'] = range(1, len(df) + 1)
    df['cumulative_pnl'] = df['pnl'].cumsum()

    print(f"✅ Processed {len(df)} trades, Total PnL: ${df['cumulative_pnl'].iloc[-1]:.2f}")
    return df

def create_pnl_chart(df, days_back):
    """Create professional PnL chart"""
    if df.empty:
        return go.Figure().add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)

    # Data for chart
    x_data = [0] + df['trade_number'].tolist()
    y_data = [0] + df['cumulative_pnl'].tolist()

    final_pnl = y_data[-1]
    total_trades = len(df)

    fig = go.Figure()

    # Main PnL line
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        line=dict(color='#10B981', width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.15)',
        hovertemplate='Trade #%{x}<br>Total P&L: $%{y:.2f}<extra></extra>'
    ))

    # Styling
    fig.add_hline(y=0, line_dash="dash", line_color="#9CA3AF", opacity=0.7)

    fig.update_layout(
        title=f'📈 Portfolio Performance<br><span style="font-size:14px; color:#6B7280">Total: ${final_pnl:.2f} | {days_back} Days | {total_trades} Trades</span>',
        plot_bgcolor='white',
        paper_bgcolor='#F9FAFB',
        xaxis=dict(title='Trade Number', showgrid=True, gridcolor='#E5E7EB'),
        yaxis=dict(title='Cumulative P&L ($)', showgrid=True, gridcolor='#E5E7EB', tickformat='$,.2f'),
        height=550,
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=60)
    )

    return fig

def create_btc_compare_pnl_chart(df, days_back):
    """Create comparison chart between PnL and BTC performance - same format as portfolio chart"""
    if df.empty:
        return go.Figure().add_annotation(text="No PnL data", x=0.5, y=0.5, showarrow=False)

    # Get the exact date range from the portfolio
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()

    print(f"🔍 Portfolio period: {start_date} to {end_date}")

    # Fetch BTC data for the exact same period
    btc_data = fetch_btc_data(start_date, end_date)
    if btc_data is None or btc_data.empty:
        return go.Figure().add_annotation(text="No BTC data available for portfolio period", x=0.5, y=0.5, showarrow=False)

    fig = go.Figure()

    # Portfolio data - same as original PnL chart (trade numbers and dollar values)
    pnl_x_data = [0] + df['trade_number'].tolist()
    pnl_y_data = [0] + df['cumulative_pnl'].tolist()

    # BTC data - calculate what $1000 invested at start would be worth
    btc_data_sorted = btc_data.sort_index()
    first_btc_price = btc_data_sorted['close'].iloc[0]

    # Assume $1000 initial investment in BTC at start of trading period
    initial_investment = 1000
    btc_quantity = initial_investment / first_btc_price

    # Calculate BTC portfolio value over time, mapped to trade numbers
    btc_values = []
    trade_timestamps = df['timestamp'].tolist()

    # Start with $1000 at trade 0
    btc_values.append(initial_investment)

    # For each trade, find the nearest BTC price and calculate portfolio value
    for trade_time in trade_timestamps:
        # Find the BTC price closest to this trade time
        time_diffs = abs(btc_data_sorted.index - trade_time)
        closest_idx = time_diffs.argmin()  # Use argmin instead of idxmin
        closest_time = btc_data_sorted.index[closest_idx]
        btc_price_at_trade = btc_data_sorted.loc[closest_time, 'close']
        btc_portfolio_value = btc_quantity * btc_price_at_trade
        btc_values.append(btc_portfolio_value)

    # Convert BTC values to P&L (relative to initial $1000)
    btc_pnl = [value - initial_investment for value in btc_values]
    btc_x_data = list(range(len(btc_pnl)))

    # Add Portfolio trace
    fig.add_trace(go.Scatter(
        x=pnl_x_data,
        y=pnl_y_data,
        mode='lines',
        name='Your Portfolio',
        line=dict(color='#10B981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.15)',
        hovertemplate='Trade #%{x}<br>Portfolio P&L: $%{y:.2f}<extra></extra>'
    ))

    # Add BTC HODL trace
    fig.add_trace(go.Scatter(
        x=btc_x_data,
        y=btc_pnl,
        mode='lines',
        name='BTC HODL ($1000)',
        line=dict(color='#F59E0B', width=2, dash='dash'),
        hovertemplate='Trade #%{x}<br>BTC P&L: $%{y:.2f}<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="#9CA3AF", opacity=0.7)

    # Calculate final values for title
    final_portfolio_pnl = pnl_y_data[-1] if pnl_y_data else 0
    final_btc_pnl = btc_pnl[-1] if btc_pnl else 0

    # Determine outperformance in dollars
    outperform = final_portfolio_pnl - final_btc_pnl
    outperform_text = f"📈 +${outperform:.2f}" if outperform > 0 else f"📉 ${outperform:.2f}"

    # Calculate actual days between first and last trade
    actual_days = (end_date - start_date).days
    total_trades = len(df)

    fig.update_layout(
        title=f'🚀 Portfolio vs BTC Comparison ({actual_days} Days)<br><span style="font-size:14px; color:#6B7280">Your P&L: ${final_portfolio_pnl:.2f} | BTC P&L: ${final_btc_pnl:.2f} | Outperformance: {outperform_text}</span>',
        plot_bgcolor='white',
        paper_bgcolor='#F9FAFB',
        xaxis=dict(
            title='Trade Number',
            showgrid=True,
            gridcolor='#E5E7EB'
        ),
        yaxis=dict(
            title='Cumulative P&L ($)',
            showgrid=True,
            gridcolor='#E5E7EB',
            tickformat='$,.2f'
        ),
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        hovermode='x unified'
    )

    return fig

def create_percentage_comparison_chart(df, days_back):
    """Create percentage returns comparison chart between portfolio and BTC"""
    if df.empty:
        return go.Figure().add_annotation(text="No PnL data", x=0.5, y=0.5, showarrow=False)

    # Get the exact date range from the portfolio
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()

    print(f"🔍 Percentage comparison for period: {start_date} to {end_date}")

    # Fetch BTC data for the exact same period
    btc_data = fetch_btc_data(start_date, end_date)
    if btc_data is None or btc_data.empty:
        return go.Figure().add_annotation(text="No BTC data available for portfolio period", x=0.5, y=0.5, showarrow=False)

    fig = go.Figure()

    # Portfolio percentage returns calculation
    # Assume starting capital (can be estimated from first few trades or use fixed amount)
    starting_capital = 1000  # Assume $1000 starting capital

    # Calculate portfolio value over time
    portfolio_values = [starting_capital]  # Start with initial capital
    for pnl in df['cumulative_pnl']:
        portfolio_values.append(starting_capital + pnl)

    # Calculate portfolio percentage returns
    portfolio_pct_returns = []
    for value in portfolio_values:
        pct_return = ((value / starting_capital) - 1) * 100
        portfolio_pct_returns.append(pct_return)

    # Portfolio X-axis (trade numbers)
    portfolio_x_data = list(range(len(portfolio_pct_returns)))

    # BTC percentage returns calculation
    btc_data_sorted = btc_data.sort_index()
    first_btc_price = btc_data_sorted['close'].iloc[0]

    # Calculate BTC percentage returns over time, mapped to trade times
    btc_pct_returns = [0]  # Start at 0%
    trade_timestamps = df['timestamp'].tolist()

    for trade_time in trade_timestamps:
        # Find the BTC price closest to this trade time
        time_diffs = abs(btc_data_sorted.index - trade_time)
        closest_idx = time_diffs.argmin()
        closest_time = btc_data_sorted.index[closest_idx]
        btc_price_at_trade = btc_data_sorted.loc[closest_time, 'close']

        # Calculate percentage return from start
        btc_pct_return = ((btc_price_at_trade / first_btc_price) - 1) * 100
        btc_pct_returns.append(btc_pct_return)

    # BTC X-axis (same as portfolio)
    btc_x_data = list(range(len(btc_pct_returns)))

    # Add Portfolio trace
    fig.add_trace(go.Scatter(
        x=portfolio_x_data,
        y=portfolio_pct_returns,
        mode='lines+markers',
        name='Your Portfolio',
        line=dict(color='#10B981', width=3),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)',
        hovertemplate='Trade #%{x}<br>Portfolio Return: %{y:.2f}%<extra></extra>'
    ))

    # Add BTC HODL trace
    fig.add_trace(go.Scatter(
        x=btc_x_data,
        y=btc_pct_returns,
        mode='lines',
        name='BTC HODL',
        line=dict(color='#F59E0B', width=2, dash='dash'),
        hovertemplate='Trade #%{x}<br>BTC Return: %{y:.2f}%<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="#9CA3AF", opacity=0.7)

    # Calculate final values for title
    final_portfolio_pct = portfolio_pct_returns[-1] if portfolio_pct_returns else 0
    final_btc_pct = btc_pct_returns[-1] if btc_pct_returns else 0

    # Determine outperformance in percentage points
    outperform = final_portfolio_pct - final_btc_pct
    outperform_text = f"📈 +{outperform:.1f}pp" if outperform > 0 else f"📉 {outperform:.1f}pp"

    # Calculate actual days between first and last trade
    actual_days = (end_date - start_date).days
    total_trades = len(df)

    # Color for performance indicators
    portfolio_color = "#10B981" if final_portfolio_pct >= 0 else "#EF4444"
    btc_color = "#F59E0B" if final_btc_pct >= 0 else "#EF4444"

    fig.update_layout(
        title=f'📊 Percentage Returns Comparison ({actual_days} Days)<br><span style="font-size:14px; color:#6B7280"><span style="color:{portfolio_color}">Portfolio: {final_portfolio_pct:+.1f}%</span> | <span style="color:{btc_color}">BTC: {final_btc_pct:+.1f}%</span> | Outperformance: {outperform_text}</span>',
        plot_bgcolor='white',
        paper_bgcolor='#F9FAFB',
        xaxis=dict(
            title='Trade Number',
            showgrid=True,
            gridcolor='#E5E7EB'
        ),
        yaxis=dict(
            title='Return (%)',
            showgrid=True,
            gridcolor='#E5E7EB',
            tickformat='.1f',
            ticksuffix='%',
            zeroline=True,
            zerolinecolor='#9CA3AF',
            zerolinewidth=1
        ),
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        hovermode='x unified'
    )

    return fig

def create_percentage_comparison_drawdown_chart(df, days_back):
    """Create drawdown comparison chart between portfolio and BTC"""
    if df.empty:
        return go.Figure().add_annotation(text="No PnL data", x=0.5, y=0.5, showarrow=False)

    # Get the exact date range from the portfolio
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()

    print(f"🔍 Drawdown comparison for period: {start_date} to {end_date}")

    # Fetch BTC data for the exact same period
    btc_data = fetch_btc_data(start_date, end_date)
    if btc_data is None or btc_data.empty:
        return go.Figure().add_annotation(text="No BTC data available for portfolio period", x=0.5, y=0.5, showarrow=False)

    fig = go.Figure()

    # Portfolio drawdown calculation
    # Assume starting capital
    starting_capital = 1000

    # Calculate portfolio value over time
    portfolio_values = [starting_capital]  # Start with initial capital
    for pnl in df['cumulative_pnl']:
        portfolio_values.append(starting_capital + pnl)

    # Calculate portfolio drawdown
    portfolio_peak = [portfolio_values[0]]  # Track running maximum
    portfolio_drawdown = [0]  # Start with 0% drawdown

    for i, value in enumerate(portfolio_values[1:], 1):
        # Update peak (running maximum)
        current_peak = max(portfolio_peak[-1], value)
        portfolio_peak.append(current_peak)

        # Calculate drawdown as percentage from peak
        drawdown_pct = ((value - current_peak) / current_peak) * 100 if current_peak > 0 else 0
        portfolio_drawdown.append(drawdown_pct)

    # Portfolio X-axis (trade numbers)
    portfolio_x_data = list(range(len(portfolio_drawdown)))

    # BTC drawdown calculation
    btc_data_sorted = btc_data.sort_index()

    # Calculate BTC drawdowns over time, mapped to trade times
    btc_values = []
    trade_timestamps = [start_date] + df['timestamp'].tolist()  # Include start

    for trade_time in trade_timestamps:
        # Find the BTC price closest to this trade time
        time_diffs = abs(btc_data_sorted.index - trade_time)
        closest_idx = time_diffs.argmin()
        closest_time = btc_data_sorted.index[closest_idx]
        btc_price_at_trade = btc_data_sorted.loc[closest_time, 'close']
        btc_values.append(btc_price_at_trade)

    # Calculate BTC drawdown
    btc_peak = [btc_values[0]]  # Track running maximum
    btc_drawdown = [0]  # Start with 0% drawdown

    for i, price in enumerate(btc_values[1:], 1):
        # Update peak (running maximum)
        current_peak = max(btc_peak[-1], price)
        btc_peak.append(current_peak)

        # Calculate drawdown as percentage from peak
        drawdown_pct = ((price - current_peak) / current_peak) * 100 if current_peak > 0 else 0
        btc_drawdown.append(drawdown_pct)

    # BTC X-axis (same as portfolio)
    btc_x_data = list(range(len(btc_drawdown)))

    # Add Portfolio drawdown trace
    fig.add_trace(go.Scatter(
        x=portfolio_x_data,
        y=portfolio_drawdown,
        mode='lines',
        name='Your Portfolio DD',
        line=dict(color='#EF4444', width=3),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)',
        hovertemplate='Trade #%{x}<br>Portfolio Drawdown: %{y:.2f}%<extra></extra>'
    ))

    # Add BTC drawdown trace
    fig.add_trace(go.Scatter(
        x=btc_x_data,
        y=btc_drawdown,
        mode='lines',
        name='BTC HODL DD',
        line=dict(color='#F59E0B', width=2, dash='dash'),
        fill='tozeroy',
        fillcolor='rgba(245, 158, 11, 0.1)',
        hovertemplate='Trade #%{x}<br>BTC Drawdown: %{y:.2f}%<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="#9CA3AF", opacity=0.7)

    # Calculate max drawdowns for title
    max_portfolio_dd = min(portfolio_drawdown) if portfolio_drawdown else 0
    max_btc_dd = min(btc_drawdown) if btc_drawdown else 0

    # Determine which had better drawdown control
    dd_comparison = max_portfolio_dd - max_btc_dd
    if dd_comparison > 0:  # Portfolio had worse drawdown
        dd_text = f"📉 {abs(dd_comparison):.1f}pp worse"
        dd_color = "text-danger"
    elif dd_comparison < 0:  # Portfolio had better drawdown control
        dd_text = f"📈 {abs(dd_comparison):.1f}pp better"
        dd_color = "text-success"
    else:
        dd_text = "📊 Similar"
        dd_color = "text-secondary"

    # Calculate actual days between first and last trade
    actual_days = (end_date - start_date).days

    fig.update_layout(
        title=f'📉 Drawdown Comparison ({actual_days} Days)<br><span style="font-size:14px; color:#6B7280">Your Max DD: {max_portfolio_dd:.1f}% | BTC Max DD: {max_btc_dd:.1f}% | Difference: {dd_text}</span>',
        plot_bgcolor='white',
        paper_bgcolor='#F9FAFB',
        xaxis=dict(
            title='Trade Number',
            showgrid=True,
            gridcolor='#E5E7EB'
        ),
        yaxis=dict(
            title='Drawdown (%)',
            showgrid=True,
            gridcolor='#E5E7EB',
            tickformat='.1f',
            ticksuffix='%',
            range=[min(min(portfolio_drawdown), min(btc_drawdown)) - 2, 2]  # Set range to show drawdowns properly
        ),
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        hovermode='x unified'
    )

    return fig

def get_stats(df):
    """Get comprehensive trading statistics"""
    if df.empty:
        return {
            'total_pnl': 0, 'win_rate': 0, 'total_trades': 0,
            'profitable_trades': 0, 'max_drawdown': 0, 'avg_daily_pnl': 0,
            'losing_trades': 0, 'largest_win': 0, 'largest_loss': 0,
            'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
            'sharpe_ratio': 0, 'btc_sharpe_ratio': 0, 'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
            'total_volume': 0, 'avg_trade_size': 0, 'trading_days': 0,
            'trades_per_day': 0, 'recovery_factor': 0, 'return_pct': 0, 'reward_risk_ratio': 0,
            'strategy_start_date': 'N/A', 'strategy_end_date': 'N/A'
        }

    # Basic metrics
    total_pnl = df['cumulative_pnl'].iloc[-1]
    win_trades = df[df['pnl'] > 0]
    loss_trades = df[df['pnl'] < 0]

    # Win/Loss metrics
    profitable_trades = len(win_trades)
    losing_trades = len(loss_trades)
    win_rate = (profitable_trades / len(df)) * 100 if len(df) > 0 else 0

    # P&L metrics
    largest_win = df['pnl'].max()
    largest_loss = df['pnl'].min()
    avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0

    # Risk metrics
    max_drawdown = (df['cumulative_pnl'] - df['cumulative_pnl'].cummax()).min()

    # Profit factor (total wins / abs(total losses))
    total_wins = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
    total_losses = abs(loss_trades['pnl'].sum()) if len(loss_trades) > 0 else 0.01  # Avoid division by zero
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    # Sharpe ratio (simplified - daily returns / std dev)
    daily_returns = df['pnl']
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0

    # Consecutive wins/losses
    def get_max_consecutive(series, condition):
        groups = (series != condition).cumsum()[series == condition]
        return groups.value_counts().max() if len(groups) > 0 else 0

    max_consecutive_wins = get_max_consecutive(df['pnl'] > 0, True)
    max_consecutive_losses = get_max_consecutive(df['pnl'] < 0, True)

    # Volume and trading frequency
    total_volume = (df['price'] * df['size']).sum()
    avg_trade_size = total_volume / len(df) if len(df) > 0 else 0

    # Time-based metrics
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    trading_days = (end_date - start_date).days
    trading_days = max(trading_days, 1)  # At least 1 day
    trades_per_day = len(df) / trading_days
    avg_daily_pnl = total_pnl / trading_days

    # Format dates for display
    strategy_start_date = start_date.strftime("%-d %b %Y")  # e.g., "1 Sep 2025"
    strategy_end_date = end_date.strftime("%-d %b %Y")      # e.g., "5 Sep 2025"

    # Recovery factor (total return / max drawdown)
    recovery_factor = abs(total_pnl / max_drawdown) if max_drawdown < 0 else 0

    # Return percentage (assume $1000 starting capital)
    starting_capital = 1000
    return_pct = (total_pnl / starting_capital) * 100

    # Reward-to-Risk Ratio (Average Win / Average Loss)
    reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss < 0 else 0

    # Calculate BTC Sharpe Ratio for comparison
    btc_sharpe_ratio = 0
    try:
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        btc_data = fetch_btc_data(start_date, end_date)

        if btc_data is not None and not btc_data.empty:
            btc_data_sorted = btc_data.sort_index()
            # Calculate BTC daily returns
            btc_prices = btc_data_sorted['close'].values
            btc_daily_returns = []

            for i in range(1, len(btc_prices)):
                daily_return = (btc_prices[i] - btc_prices[i-1]) / btc_prices[i-1]
                btc_daily_returns.append(daily_return)

            if len(btc_daily_returns) > 1:
                btc_daily_returns = pd.Series(btc_daily_returns)
                btc_sharpe_ratio = (btc_daily_returns.mean() / btc_daily_returns.std()) * (252 ** 0.5) if btc_daily_returns.std() > 0 else 0

    except Exception as e:
        print(f"Error calculating BTC Sharpe Ratio: {e}")
        btc_sharpe_ratio = 0

    return {
        # Basic metrics
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'total_trades': len(df),
        'profitable_trades': profitable_trades,
        'losing_trades': losing_trades,
        'max_drawdown': max_drawdown,
        'avg_daily_pnl': avg_daily_pnl,

        # P&L metrics
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,

        # Risk metrics
        'sharpe_ratio': sharpe_ratio,
        'btc_sharpe_ratio': btc_sharpe_ratio,
        'recovery_factor': recovery_factor,
        'return_pct': return_pct,
        'reward_risk_ratio': reward_risk_ratio,

        # Streak metrics
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,

        # Volume metrics
        'total_volume': total_volume,
        'avg_trade_size': avg_trade_size,

        # Frequency metrics
        'trading_days': trading_days,
        'trades_per_day': trades_per_day,
        'strategy_start_date': strategy_start_date,
        'strategy_end_date': strategy_end_date
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, user_address: str = Form(...), days_back: int = Form(56)):
    try:
        # Fetch and process data
        raw_data = fetch_hyperliquid_data(user_address, days_back)
        df = process_data(raw_data)

        if df.empty:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "No trading data found"
            })

        # Create all charts
        pnl_chart = create_pnl_chart(df, days_back)
        btc_compare_chart = create_btc_compare_pnl_chart(df, days_back)
        percentage_chart = create_percentage_comparison_chart(df, days_back)
        drawdown_chart = create_percentage_comparison_drawdown_chart(df, days_back)
        stats = get_stats(df)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "chart": json.dumps(pnl_chart, cls=plotly.utils.PlotlyJSONEncoder),
            "btc_chart": json.dumps(btc_compare_chart, cls=plotly.utils.PlotlyJSONEncoder),
            "percentage_chart": json.dumps(percentage_chart, cls=plotly.utils.PlotlyJSONEncoder),
            "drawdown_chart": json.dumps(drawdown_chart, cls=plotly.utils.PlotlyJSONEncoder),
            "stats": stats,
            "user_address": user_address,
            "days_back": days_back
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
