#!/bin/bash
# Script to launch Ting ling ling in the background
cd "/Users/nishkarsh/Documents/Projectssss/slM Small - Ting ling ling"

# Check if it's already running on port 5002 (we changed the port earlier)
PID=$(lsof -ti:5002)
if [ ! -z "$PID" ]; then
    kill -9 $PID
fi

# Run the app in the background, logging to tinglingling.log
export PYTHONUNBUFFERED=1
nohup .venv/bin/python3 app.py > tinglingling.log 2>&1 &

