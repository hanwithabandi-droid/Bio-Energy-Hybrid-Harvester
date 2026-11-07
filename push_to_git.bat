@echo off
setlocal enabledelayedexpansion

echo Syncing repository with GitHub...
echo Repository: https://github.com/hanwithabandi-droid/Bio-Energy-Hybrid-Harvester

set "GIT_CMD=git"
where git >nul 2>nul
if %errorlevel% neq 0 (
    for /d %%D in ("%LOCALAPPDATA%\GitHubDesktop\app-*") do (
        if exist "%%D\resources\app\git\cmd\git.exe" (
            set "GIT_CMD=%%D\resources\app\git\cmd\git.exe"
        )
    )
)

set "MSG=%~1"
if "%MSG%"=="" set "MSG=Bio-energy hybrid harvester simulation and documentation"

"%GIT_CMD%" add -A
"%GIT_CMD%" commit -m "%MSG%"
"%GIT_CMD%" push origin main

if %errorlevel% equ 0 (
    echo.
    echo Pushed successfully to:
    echo https://github.com/hanwithabandi-droid/Bio-Energy-Hybrid-Harvester
) else (
    echo.
    echo Push failed. Check your network connection or GitHub Desktop credentials.
)

pause
