@echo off
setlocal EnableDelayedExpansion

set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
set VENV_DIR=%~dp0.venv
set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe

:: ============================================================
:: PYTHON 3.10.11
:: ============================================================
echo.
echo [1/4] Checking Python 3.10.11...

if exist "!PYTHON_EXE!" (
    for /f "tokens=2" %%V in ('"!PYTHON_EXE!" --version 2^>^&1') do set PYVER=%%V
    if "!PYVER!" == "3.10.11" (
        echo  ^> Python 3.10.11 already installed. Skipping.
        goto :venv
    )
)

echo  ^> Downloading Python 3.10.11...
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe' -OutFile '%TEMP%\python-3.10.11.exe'"
if %errorlevel% neq 0 (
    echo FAILED: Could not download Python 3.10.11.
    pause
    exit /b 1
)

echo  ^> Installing Python 3.10.11...
"%TEMP%\python-3.10.11.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0
if %errorlevel% neq 0 (
    echo FAILED: Python installation failed.
    pause
    exit /b 1
)

if not exist "!PYTHON_EXE!" (
    echo FAILED: Python installed but not found at: !PYTHON_EXE!
    pause
    exit /b 1
)
echo  ^> Python 3.10.11 installed successfully.


:: ============================================================
:: DOWNLOAD NICESHOT_AI
:: ============================================================
echo.
echo Downloading NiceShot_AI...

set REPO_ZIP=%TEMP%\NiceShot_AI.zip
set REPO_URL=https://github.com/karimm-ai/NiceShot_AI/archive/refs/heads/main.zip
set INSTALL_DIR=%~dp0NiceShot_AI

:: Skip if already downloaded
if exist "!INSTALL_DIR!\niceshot_ai.py" (
    echo  ^> NiceShot_AI already exists. Skipping.
    goto :download_done
)

powershell -NoProfile -Command "Invoke-WebRequest -Uri '!REPO_URL!' -OutFile '!REPO_ZIP!'"
if %errorlevel% neq 0 (
    echo FAILED: Could not download NiceShot_AI. Check your internet connection.
    pause
    exit /b 1
)

:: Extract and flatten — GitHub zips have a versioned subfolder inside e.g. NiceShot_AI-main\
powershell -NoProfile -Command "Expand-Archive -Path '!REPO_ZIP!' -DestinationPath '%TEMP%\NiceShot_AI_extracted' -Force"
if %errorlevel% neq 0 (
    echo FAILED: Could not extract NiceShot_AI.
    pause
    exit /b 1
)

:: Move the inner folder to final location
powershell -NoProfile -Command "Move-Item -Path '%TEMP%\NiceShot_AI_extracted\NiceShot_AI-main' -Destination '!INSTALL_DIR!' -Force"
if %errorlevel% neq 0 (
    echo FAILED: Could not move NiceShot_AI to install directory.
    pause
    exit /b 1
)

echo  ^> NiceShot_AI downloaded successfully.

:download_done


:: ============================================================
:: VIRTUAL ENVIRONMENT
:: ============================================================
:venv
echo.
echo [2/4] Setting up virtual environment...

if exist "!VENV_PYTHON!" (
    echo  ^> Virtual environment already exists. Skipping.
    goto :requirements
)

"!PYTHON_EXE!" -m venv "!VENV_DIR!"
if %errorlevel% neq 0 (
    echo FAILED: Could not create virtual environment.
    pause
    exit /b 1
)
echo  ^> Virtual environment created at: !VENV_DIR!


:: ============================================================
:: REQUIREMENTS.TXT
:: ============================================================
:requirements
echo.
echo [3/4] Installing requirements...

if not exist "%~dp0requirements.txt" (
    echo FAILED: requirements.txt not found next to install.bat.
    pause
    exit /b 1
)

"!VENV_PYTHON!" -m pip install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    echo FAILED: Could not install requirements.txt.
    pause
    exit /b 1
)
echo  ^> Requirements installed successfully.


:: ============================================================
:: TORCH
:: ============================================================
:torch
echo.
echo Installing PyTorch...

:: Check if correct torch is already installed
"!VENV_PYTHON!" -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if %errorlevel% == 0 (
    echo  ^> PyTorch with CUDA already installed. Skipping.
    goto :verify
)
"!VENV_PYTHON!" -c "import torch" >nul 2>&1
if %errorlevel% == 0 (
    echo  ^> PyTorch found but CUDA not available. Reinstalling with CUDA support...
    "!VENV_PYTHON!" -m pip uninstall torch torchvision torchaudio -y >nul 2>&1
)

:: Pre-install typing-extensions from PyPI to avoid PyTorch index naming bug
echo  ^> Pre-installing typing-extensions...
"!VENV_PYTHON!" -m pip install "typing-extensions>=4.10.0" >nul 2>&1

:: Detect CUDA version
set TORCH_INDEX=https://download.pytorch.org/whl/cpu
set TORCH_NIGHTLY=0
set CUDA_FOUND=0

where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 goto :no_gpu

for /f "tokens=*" %%L in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set FULL_LINE=%%L
for /f "tokens=2 delims=:" %%A in ("!FULL_LINE:*CUDA Version=CUDA Version!") do set CUDA_RAW=%%A
for /f "tokens=1" %%V in ("!CUDA_RAW!") do set CUDA_VER=%%V
for /f "tokens=1,2 delims=." %%A in ("!CUDA_VER!") do (
    set CUDA_MAJOR=%%A
    set CUDA_MINOR=%%B
)
set /a CUDA_NUM=CUDA_MAJOR*10+CUDA_MINOR
set CUDA_FOUND=1
echo  ^> NVIDIA GPU detected. CUDA !CUDA_MAJOR!.!CUDA_MINOR!

:: Write a small helper script to get compute capability — avoids quote issues in for/f
echo import subprocess > "%TEMP%\get_cap.py"
echo o = subprocess.check_output(["nvidia-smi","--query-gpu=compute_cap","--format=csv,noheader"]).decode().strip() >> "%TEMP%\get_cap.py"
echo print(o.replace(".","")) >> "%TEMP%\get_cap.py"

"!VENV_PYTHON!" "%TEMP%\get_cap.py" > "%TEMP%\compute_cap.txt" 2>nul
set /p COMPUTE_CAP=<"%TEMP%\compute_cap.txt"
echo  ^> GPU compute capability: !COMPUTE_CAP!

:: Blackwell (sm_120+) requires nightly build
if !COMPUTE_CAP! geq 120 (
    echo  ^> Blackwell GPU detected. Stable PyTorch does not support sm_!COMPUTE_CAP!. Using nightly build.
    set TORCH_NIGHTLY=1
    set TORCH_INDEX=https://download.pytorch.org/whl/nightly/cu128
    goto :torch_install
)

:: Map CUDA version to stable torch build
if !CUDA_NUM! geq 126 ( set TORCH_INDEX=https://download.pytorch.org/whl/cu126 & goto :torch_install )
if !CUDA_NUM! geq 124 ( set TORCH_INDEX=https://download.pytorch.org/whl/cu124 & goto :torch_install )
if !CUDA_NUM! geq 121 ( set TORCH_INDEX=https://download.pytorch.org/whl/cu121 & goto :torch_install )
if !CUDA_NUM! geq 118 ( set TORCH_INDEX=https://download.pytorch.org/whl/cu118 & goto :torch_install )

echo  ^> CUDA !CUDA_VER! is too old (minimum 11.8). Falling back to CPU.

:no_gpu
if !CUDA_FOUND! == 0 (
    echo  ^> No NVIDIA GPU detected. Installing CPU-only PyTorch.
    echo  ^> WARNING: Inference will be significantly slower without a GPU.
)
set TORCH_INDEX=https://download.pytorch.org/whl/cpu

:torch_install
echo  ^> Installing PyTorch from: !TORCH_INDEX!
if !TORCH_NIGHTLY! == 1 (
    "!VENV_PYTHON!" -m pip install --pre torch torchvision torchaudio --index-url !TORCH_INDEX!
) else (
    "!VENV_PYTHON!" -m pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url !TORCH_INDEX!
)
if %errorlevel% neq 0 (
    echo FAILED: PyTorch installation failed.
    pause
    exit /b 1
)
echo  ^> PyTorch installed successfully.

:verify
echo.
echo  ^> Verifying installation...
"!VENV_PYTHON!" -c "import torch; print('  Torch version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
echo.
pause


:: ============================================================
:: DONE
:: ============================================================
:done
echo.
echo  ==============================================
echo   Installation complete!
echo   Python executable for your app to use:
echo   !VENV_PYTHON!
echo  ==============================================
echo.
pause
