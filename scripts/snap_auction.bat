@echo off
rem Wrapper for Windows Task Scheduler. ASCII only: cmd parses this file before chcp runs.
chcp 65001 >nul
cd /d "%~dp0.."
if not exist logs mkdir logs
python -X utf8 scripts\snap_auction.py >> logs\snap_auction.log 2>&1
