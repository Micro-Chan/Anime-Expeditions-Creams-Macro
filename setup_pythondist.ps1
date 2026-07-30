# Sets up PythonDist\<version> -- a self-contained, Windows-embeddable Python
# with pip and every core/requirements.txt dependency pre-installed, so the
# app runs without touching whatever Python (if any) is on the user's PATH.
# Called by run.bat only when PythonDist is missing; safe to re-run by hand
# if a dependency install gets interrupted partway.
#
# --no-build-isolation on the requirements.txt install (after pre-installing
# setuptools/wheel into the dist itself) works around a real embeddable-
# Python limitation: the python312._pth file this script writes makes
# CPython ignore PYTHONPATH entirely, which is exactly the mechanism pip's
# isolated build environments rely on -- so any dependency needing a source
# build (no prebuilt wheel for this Python/platform) fails to see its build
# backend unless isolation is turned off and the build tools already live in
# the dist's own site-packages instead.
$ErrorActionPreference = "Stop"

$PyVersion = "3.12.10"
$DistDir = Join-Path $PSScriptRoot "PythonDist\python$PyVersion"
$ZipUrl = "https://www.python.org/ftp/python/$PyVersion/python-$PyVersion-embed-amd64.zip"
$GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
$PthFile = Join-Path $DistDir "python312._pth"
$PythonExe = Join-Path $DistDir "python.exe"
$RequirementsFile = Join-Path $PSScriptRoot "requirements.txt"

if (Test-Path $PythonExe) {
    Write-Host "PythonDist\python$PyVersion already set up."
    exit 0
}

Write-Host "Setting up PythonDist\python$PyVersion (first run only) ..."
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

$ZipPath = Join-Path $env:TEMP "python-$PyVersion-embed-amd64.zip"
Write-Host "Downloading embeddable Python $PyVersion..."
Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath
Expand-Archive -Path $ZipPath -DestinationPath $DistDir -Force
Remove-Item $ZipPath -Force

# Embeddable Python ships with site-packages/PYTHONPATH disabled by default
# (a bare "#import site" line) -- re-enable it and add Lib\site-packages
# (where pip installs to) plus the repo root (..\..\ from PythonDist\pyX.Y),
# so `import core...` resolves when main.py runs from here.
@"
python312.zip
.
Lib\site-packages
..\..\

import site
"@ | Set-Content -Path $PthFile -Encoding ascii

Write-Host "Bootstrapping pip..."
$GetPipPath = Join-Path $DistDir "get-pip.py"
Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipPath
& $PythonExe $GetPipPath
if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed (exit $LASTEXITCODE)" }

Write-Host "Installing build tools (setuptools, wheel)..."
& $PythonExe -m pip install setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "Installing setuptools/wheel failed (exit $LASTEXITCODE)" }

Write-Host "Installing requirements.txt..."
& $PythonExe -m pip install -r $RequirementsFile --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed (exit $LASTEXITCODE)" }

Write-Host "PythonDist\python$PyVersion is ready."
