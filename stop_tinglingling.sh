#!/bin/bash
# Script to stop Ting ling ling
PID=$(lsof -ti:5002)
if [ ! -z "$PID" ]; then
    kill -9 $PID
    killall say 2>/dev/null
    osascript -e 'display notification "Ting ling ling has been shut down." with title "Ting ling ling"'
else
    osascript -e 'display notification "Ting ling ling is not currently running." with title "Ting ling ling"'
fi
