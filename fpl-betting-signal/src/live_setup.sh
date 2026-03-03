#!/bin/bash
# Setup script for FPL Live Tracker
# Run once to create directories and test connectivity

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LIVE_DIR="$PROJECT_DIR/data/live/2025-26"

echo "============================================"
echo "FPL Live Tracker Setup"
echo "============================================"

# 1. Create directories
echo ""
echo "[1/4] Creating directories..."
mkdir -p "$LIVE_DIR/snapshots"
echo "  Created $LIVE_DIR/snapshots"

# 2. Check Python dependencies
echo ""
echo "[2/4] Checking Python dependencies..."
python3 -c "
import pandas, numpy, sklearn, requests, matplotlib
print('  All dependencies OK')
" 2>/dev/null || {
    echo "  Installing dependencies..."
    pip install pandas numpy scikit-learn requests matplotlib pyarrow
}

# 3. Test FPL API connectivity
echo ""
echo "[3/4] Testing FPL API connectivity..."
python3 -c "
import requests
try:
    resp = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/',
                       timeout=15, headers={'User-Agent': 'FPL-Tracker/1.0'})
    if resp.status_code == 200:
        data = resp.json()
        events = data['events']
        current = [e for e in events if e.get('is_current')]
        gw = current[0]['id'] if current else 'unknown'
        print(f'  FPL API: OK (Current GW: {gw})')
        print(f'  Players: {len(data[\"elements\"])}')
        print(f'  Teams: {len(data[\"teams\"])}')

        # Show next deadline
        upcoming = [e for e in events if not e['finished']]
        if upcoming:
            dl = upcoming[0]['deadline_time']
            print(f'  Next deadline: {dl}')
    else:
        print(f'  FPL API: HTTP {resp.status_code}')
except Exception as e:
    print(f'  FPL API: FAILED ({e})')
    print('  NOTE: The FPL API must be accessible from your network.')
    print('  If behind a proxy, ensure fantasy.premierleague.com is whitelisted.')
"

# 4. Test model availability
echo ""
echo "[4/4] Checking pre-trained models..."
python3 -c "
import pickle, os
model_dir = '$PROJECT_DIR/outputs/model'
models = ['model_fpl_only.pkl', 'model_odds_only.pkl', 'model_fpl_odds.pkl', 'scaler.pkl']
all_ok = True
for m in models:
    path = os.path.join(model_dir, m)
    if os.path.exists(path):
        print(f'  {m}: OK ({os.path.getsize(path):,} bytes)')
    else:
        print(f'  {m}: MISSING')
        all_ok = False
if all_ok:
    print('  All models ready for live prediction.')
else:
    print('  WARNING: Some models missing. Run the training pipeline first:')
    print('    python src/04_model_train.py')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Usage:"
echo "  # Before each GW deadline (Thursday evening):"
echo "  python3 $SCRIPT_DIR/live_tracker.py snapshot"
echo ""
echo "  # After matches complete (Sunday/Monday):"
echo "  python3 $SCRIPT_DIR/live_tracker.py results"
echo ""
echo "  # View season scorecard:"
echo "  python3 $SCRIPT_DIR/live_tracker.py scorecard"
echo ""
echo "  # Full update (snapshot + results + scorecard):"
echo "  python3 $SCRIPT_DIR/live_tracker.py update"
echo ""
echo "Automated scheduling (crontab -e):"
echo "  # Snapshot every Thursday at 11:00 PM UTC (before FPL deadline)"
echo "  0 23 * * 4 cd $PROJECT_DIR && python3 src/live_tracker.py snapshot >> data/live/2025-26/cron.log 2>&1"
echo ""
echo "  # Record results every Monday at 8:00 AM UTC"
echo "  0 8 * * 1 cd $PROJECT_DIR && python3 src/live_tracker.py results >> data/live/2025-26/cron.log 2>&1"
echo "============================================"
