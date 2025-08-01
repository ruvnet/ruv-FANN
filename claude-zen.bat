@echo off
:: Claude Zen CLI for Windows
:: AI-Driven Development Toolkit

setlocal

:: Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    exit /b 1
)

:: Run Claude Zen CLI
node "%~dp0claude-zen" %*

endlocal