@echo off
chcp 65001 >nul
pushd "%~dp0"
"flybrain\.venv\Scripts\python.exe" -B -X utf8 -u workflow.py serve --open --port 8774
popd
pause
