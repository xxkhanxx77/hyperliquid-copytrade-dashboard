# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this PnL Dashboard project.

## Project Overview

This is a reusable P&L (Profit & Loss) analytics dashboard built with FastAPI, Plotly, Pandas, and NumPy. The application provides interactive visualization and analysis of trading/investment performance data, with integrated support for Hyperliquid API data fetching.

## Architecture

### Core Components

**Backend Framework**: FastAPI with Jinja2 templates
**Data Processing**: Pandas and NumPy for data manipulation and analysis
**Visualization**: Plotly for interactive charts and graphs  
**Data Sources**: Hyperliquid API integration with HyperliquidClient
**Server**: Uvicorn ASGI server with hot reload support
**Dependency Management**: UV for fast Python package management

### Project Structure

```
pnl-dashboard/
├── src/
│   └── pnl_dashboard/
│       ├── main.py          # FastAPI application and PnL analyzer
│       └── hyperliquid_client.py  # Hyperliquid API client
├── templates/
│   ├── base.html           # Base template with navigation
│   ├── index.html          # Data source selection page
│   └── dashboard.html      # Analytics dashboard
├── static/                 # Static assets (CSS, JS, images)
├── run.py                  # Development server runner
├── pyproject.toml          # Project configuration and dependencies
└── CLAUDE.md              # This documentation file
```

## Key Features

### PnLAnalyzer Class
- **Data Processing**: Automatically prepares timestamp data and calculates cumulative P&L
- **Chart Generation**: Creates interactive Plotly visualizations
- **Statistical Analysis**: Computes key performance metrics

### Dashboard Capabilities
- **File Upload**: CSV data upload with validation
- **Sample Data**: Generate sample trading data for testing
- **Interactive Charts**: 
  - Cumulative P&L over time
  - Daily P&L distribution (bar chart)
  - Position value tracking (when available)
- **Summary Statistics**:
  - Total P&L
  - Average daily P&L
  - Win rate percentage
  - Maximum drawdown
  - Trade counts

### Data Format Requirements
CSV files should contain these columns:
- `timestamp`: Date/time in ISO format (YYYY-MM-DD HH:MM:SS)
- `pnl`: Profit/Loss amount for each trade/period
- `price` (optional): Asset price
- `quantity` (optional): Position size

## Development Commands

### Setup and Installation
```bash
# Initialize project with UV
uv init pnl-dashboard
cd pnl-dashboard

# Install dependencies
uv add fastapi uvicorn plotly pandas numpy jinja2 python-multipart
```

### Running the Application
```bash
# Development server with hot reload
python run.py

# Or directly with uvicorn
uvicorn src.pnl_dashboard.main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the Dashboard
- **URL**: http://localhost:8000
- **Upload Data**: Use the file upload form on the homepage
- **Sample Data**: Click "Use Sample Data" to test with generated data

## Code Style Guidelines

- Use FastAPI best practices with proper type hints
- Follow PEP 8 style conventions
- Use Plotly for all data visualizations
- Keep templates responsive with Bootstrap CSS
- Handle errors gracefully with user-friendly messages

## Extension Points

### Adding New Chart Types
1. Add method to `PnLAnalyzer` class (e.g., `create_risk_metrics_chart()`)
2. Update the `/upload` and `/sample-data` endpoints to include new chart
3. Add corresponding div element in `dashboard.html` template
4. Include Plotly.newPlot() call in JavaScript section

### Adding New Metrics
1. Extend `get_summary_stats()` method in `PnLAnalyzer`
2. Update `dashboard.html` template with new stat cards
3. Apply appropriate styling (positive/negative classes)

### Database Integration
- Replace CSV upload with database models
- Add SQLAlchemy or similar ORM for data persistence
- Implement data source connectors (APIs, databases)

## Security Considerations

- File upload validation and size limits
- Input sanitization for uploaded CSV data
- Consider authentication for production deployment
- HTTPS termination for production environments

## Performance Notes

- Large datasets may require pagination or data sampling
- Consider caching for frequently accessed calculations
- Use background tasks for heavy data processing
- Implement data streaming for real-time updates

## Testing Strategy

- Unit tests for `PnLAnalyzer` methods
- Integration tests for FastAPI endpoints
- Frontend testing for chart rendering
- Data validation testing with various CSV formats