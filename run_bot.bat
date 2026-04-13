@echo off
:: ============================================================
:: AI STOCK BOT — Task Scheduler entry point
::
:: INTERVAL='1h' (intraday):
::   Schedule this script every 30 min, Mon-Fri, 09:30-16:00.
::   The bot checks Alpaca's market clock and skips if closed.
::   To set up in Task Scheduler:
::     Trigger  → Daily, repeat every 30 minutes, 09:00-16:30
::     Days     → Mon Tue Wed Thu Fri only
::
:: INTERVAL='1d' (daily):
::   Schedule once per day, e.g. 16:30 Mon-Fri.
:: ============================================================

cd /d "C:\Users\labin\My Drive (labinjoayomikun@gmail.com)\AI Stock bot"
"C:\Users\labin\My Drive (labinjoayomikun@gmail.com)\AI Stock bot\.venv\Scripts\python.exe" src/run_bot.py >> data/bot_output.log 2>&1
