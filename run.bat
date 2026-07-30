@echo off
setlocal
cd /d "%~dp0"

rem Self-contained bundled Python (see setup_pythondist.ps1) instead of
rem whatever/whether Python is on PATH -- keeps this app's dependencies
rem (opencv, numpy, ...) from colliding with any other Python install on the
rem machine. PY_VERSION here must match setup_pythondist.ps1's $PyVersion.
set "PY_VERSION=3.12.10"
set "PYEXE=PythonDist\python%PY_VERSION%\python.exe"
set "PYWEXE=PythonDist\python%PY_VERSION%\pythonw.exe"

if not exist "%PYEXE%" (
    echo Bundled Python not found -- setting it up ^(first run only^)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "setup_pythondist.ps1"
    if errorlevel 1 (
        echo Failed to set up PythonDist. See the messages above for details.
        pause
        endlocal
        exit /b 1
    )
)

rem pythonw.exe (no console of its own) launched detached via Start-Process,
rem so this console isn't tied to the app's lifetime and can close as soon
rem as it's confirmed running, instead of sitting open for the whole
rem session. A short grace period first: an immediate crash (missing
rem dependency, a startup exception before main.py's own debug.log logger
rem is even up) exits well within it, and still gets caught here since a
rem real launch takes noticeably longer than this to get this far.
for /f "usebackq" %%P in (`powershell -NoProfile -Command ^
    "(Start-Process -FilePath '%PYWEXE%' -ArgumentList 'main.py' -PassThru).Id"`) do set "APP_PID=%%P"

timeout /t 3 /nobreak >nul
tasklist /fi "PID eq %APP_PID%" 2>nul | find "%APP_PID%" >nul
if errorlevel 1 (
    echo.
    echo The app closed immediately after launching -- check debug.log for details.
    pause
    endlocal
    exit /b 1
)

endlocal
