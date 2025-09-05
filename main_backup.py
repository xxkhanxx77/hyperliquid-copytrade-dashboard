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

def fetch_hyperliquid_data(user_address: str, days_back: int = 30):
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
    """Create comparison chart between PnL and BTC performance for exact same period"""
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

    # Prepare PnL data with timestamps (not trade numbers)
    df_with_zero = pd.concat([
        pd.DataFrame({'timestamp': [start_date], 'cumulative_pnl': [0]}),
        df[['timestamp', 'cumulative_pnl']]
    ]).reset_index(drop=True)

    # Calculate PnL percentage returns
    initial_balance = 1000  # Assume $1000 initial balance
    pnl_pct = (df_with_zero['cumulative_pnl'] / initial_balance) * 100

    # Prepare BTC data - calculate percentage returns from the start of portfolio period
    btc_data_sorted = btc_data.sort_index()
    first_btc_price = btc_data_sorted['close'].iloc[0]
    btc_returns = ((btc_data_sorted['close'] / first_btc_price) - 1) * 100

    # Add Portfolio trace (using timestamp)
    fig.add_trace(go.Scatter(
        x=df_with_zero['timestamp'],
        y=pnl_pct,
        mode='lines+markers',
        name='Your Portfolio',
        line=dict(color='#10B981', width=3),
        marker=dict(size=4),
        hovertemplate='%{x}<br>Portfolio Return: %{y:.2f}%<extra></extra>'
    ))

    # Add BTC trace (using timestamp)
    fig.add_trace(go.Scatter(
        x=btc_data_sorted.index,
        y=btc_returns,
        mode='lines',
        name='BTC (HODL)',
        line=dict(color='#F59E0B', width=2, dash='dash'),
        hovertemplate='%{x}<br>BTC Return: %{y:.2f}%<extra></extra>'
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="#9CA3AF", opacity=0.7)

    # Calculate final returns for title
    final_pnl_pct = pnl_pct.iloc[-1] if len(pnl_pct) > 0 else 0
    final_btc_pct = btc_returns.iloc[-1] if len(btc_returns) > 0 else 0

    # Determine outperformance
    outperform = final_pnl_pct - final_btc_pct
    outperform_text = f"📈 +{outperform:.1f}%" if outperform > 0 else f"📉 {outperform:.1f}%"

    # Calculate actual days between first and last trade
    actual_days = (end_date - start_date).days

    fig.update_layout(
        title=f'🚀 Portfolio vs BTC Comparison ({actual_days} Days)<br><span style="font-size:14px; color:#6B7280">Your Portfolio: {final_pnl_pct:.1f}% | BTC: {final_btc_pct:.1f}% | Outperformance: {outperform_text}</span>',
        plot_bgcolor='white',
        paper_bgcolor='#F9FAFB',
        xaxis=dict(
            title=f'Time ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})',
            showgrid=True,
            gridcolor='#E5E7EB',
            type='date'
        ),
        yaxis=dict(
            title='Return (%)',
            showgrid=True,
            gridcolor='#E5E7EB',
            tickformat='.1f',
            ticksuffix='%'
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
    """Get basic stats"""
    if df.empty:
        return {
            'total_pnl': 0, 'win_rate': 0, 'total_trades': 0,
            'profitable_trades': 0, 'max_drawdown': 0, 'avg_daily_pnl': 0
        }

    return {
        'total_pnl': df['cumulative_pnl'].iloc[-1],
        'win_rate': (df['pnl'] > 0).mean() * 100,
        'total_trades': len(df),
        'profitable_trades': (df['pnl'] > 0).sum(),
        'max_drawdown': (df['cumulative_pnl'] - df['cumulative_pnl'].cummax()).min(),
        'avg_daily_pnl': df['pnl'].mean()
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, user_address: str = Form(...), days_back: int = Form(30)):
    try:
        # Fetch and process data
        raw_data = fetch_hyperliquid_data(user_address, days_back)
        df = process_data(raw_data)

        if df.empty:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "No trading data found"
            })

        # Create both charts
        pnl_chart = create_pnl_chart(df, days_back)
        btc_compare_chart = create_btc_compare_pnl_chart(df, days_back)
        stats = get_stats(df)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "chart": json.dumps(pnl_chart, cls=plotly.utils.PlotlyJSONEncoder),
            "btc_chart": json.dumps(btc_compare_chart, cls=plotly.utils.PlotlyJSONEncoder),
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
