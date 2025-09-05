from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import plotly
import requests
from datetime import datetime

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

        # Create chart and get stats
        chart = create_pnl_chart(df, days_back)
        stats = get_stats(df)

        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "chart": json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder),
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
