@echo off
rem FCPO daily serving loop — run by Windows Task Scheduler (Mon-Fri 18:30 ET).
rem Appends all output to logs\serve.log for debugging failed runs.
cd /d "C:\Palm Oil"
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
echo. >> "logs\serve.log"
echo ================ %date% %time% ================ >> "logs\serve.log"
"C:\Users\megat\AppData\Local\Programs\Python\Python311\python.exe" -m src.serving.serve >> "logs\serve.log" 2>&1
echo serve exit code: %errorlevel% >> "logs\serve.log"

rem ── publish the dashboard to GitHub Pages (docs/index.html) ──────────────
"C:\Users\megat\AppData\Local\Programs\Python\Python311\python.exe" -m src.serving.dashboard --publish >> "logs\serve.log" 2>&1
git add docs/index.html >> "logs\serve.log" 2>&1
git diff --cached --quiet -- docs/index.html
if errorlevel 1 (
  git commit -m "Auto-update dashboard (scheduled run)" >> "logs\serve.log" 2>&1
  git pull --rebase --autostash >> "logs\serve.log" 2>&1
  git push >> "logs\serve.log" 2>&1
)
echo publish exit code: %errorlevel% >> "logs\serve.log"
