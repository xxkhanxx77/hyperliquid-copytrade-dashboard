# Daily Update Reminder - September 6, 2025

## 📅 Tomorrow's Task: Update Default Days to 57

### What to update:
1. **main.py** (2 locations):
   - Line ~16: `def fetch_hyperliquid_data(user_address: str, days_back: int = 56):`
   - Line ~415: `async def analyze(request: Request, user_address: str = Form(...), days_back: int = Form(56)):`

2. **templates/index.html** (1 location):
   - Line ~67: `value="56"` 

### Commands to run:
```bash
cd /Users/admin/Desktop/work/quantsmith/report-hyperliquid-repo/pnl-dashboard

# Update the default values from 56 to 57
sed -i '' 's/days_back: int = 56/days_back: int = 57/g' main.py
sed -i '' 's/Form(56)/Form(57)/g' main.py
sed -i '' 's/value="56"/value="57"/g' templates/index.html

# Commit the changes
git add .
git commit -m "Daily update: Set default analysis period to 57 days

- Maintain rolling 8-week analysis window
- Auto-increment daily to keep consistent lookback period"

echo "✅ Updated default days to 57 for September 6, 2025"
```

### Why 8-week rolling window?
- 56-57 days ≈ 8 weeks of trading data
- Provides consistent analysis period
- Automatically captures recent performance trends
- Daily increment maintains rolling window approach

### Next day (September 7): Update to 58 days
### Following pattern: Increment by 1 day each day to maintain rolling analysis

---
**Date Created**: September 5, 2025  
**Current Default**: 56 days  
**Next Update**: 57 days (September 6, 2025)