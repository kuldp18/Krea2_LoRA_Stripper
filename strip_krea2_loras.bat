@echo off
setlocal

echo ============================================
echo   Krea2 LoRA Batch Stripper
echo ============================================
echo.
echo This will scan a folder for Krea2 LoRAs and
echo create size-reduced copies (text-conditioning
echo layers only). Originals are never overwritten.
echo.

set /p LORA_PATH="Paste the full path to your LoRA folder: "

if not exist "%LORA_PATH%" (
    echo.
    echo ERROR: That path does not exist. Check it and try again.
    pause
    exit /b 1
)

echo.
echo Running on: %LORA_PATH%
echo.

python "%~dp0batch_strip_krea2.py" "%LORA_PATH%"

echo.
echo ============================================
echo Done. Press any key to close.
echo ============================================
pause >nul