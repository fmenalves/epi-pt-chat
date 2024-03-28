#!/bin/sh

# Define the directory where you want to store your logs
LOG_DIR="./logs"

# Check if the log directory exists, create it if it doesn't
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Define log file paths
ACCESS_LOG="$LOG_DIR/access.log"
ERROR_LOG="$LOG_DIR/error.log"

gunicorn run:app -w 2 --threads 2 -b 0.0.0.0:5005 --access-logfile logs/access.log --error-logfile logs/msgstore.log --log-level info 