@echo off
setlocal

echo ################################################################################
echo # Windows Setup Script for llm-social-media-cheap                            #
echo ################################################################################
echo.
echo This script attempts to replicate the setup steps from install_modules.sh
echo and install_dependencies.sh for a Windows environment.
echo.
echo IMPORTANT:
echo - This script assumes you have Python 3.12.x (as mentioned in README) and
echo   Git for Windows installed and available in your PATH.
echo - Refer to c:\Sabareesh\Vector\Jupyter\README.md for full project details and context.
echo - Some steps, especially for llama.cpp compilation and unsloth_zoo patching,
echo   might require manual intervention if the automated steps fail or if the
echo   original scripts contain more complex logic.
echo.
echo Press any key to continue or Ctrl+C to abort...
pause > nul
echo.

REM == Part 1: Equivalent of install_modules.sh ==
echo [INFO] Setting up Python virtual environment and installing requirements...
echo.

REM Check for Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.12.x and add it to PATH.
    goto :eof
)
echo [INFO] Python found.

REM Create virtual environment
if exist .venv (
    echo [INFO] Virtual environment .venv already exists. Skipping creation.
) else (
    echo [INFO] Creating Python virtual environment in .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto :eof
    )
    echo [INFO] Virtual environment created.
)
echo.

REM Install requirements
echo [INFO] Installing packages from requirements.txt...
if not exist requirements.txt (
    echo [ERROR] requirements.txt not found. Please ensure it's in the current directory.
    echo          (Current directory: %cd%)
    goto :eof
)
call .venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements. Check requirements.txt and your internet connection.
    echo          The README mentions specific library versions are important.
    goto :eof
)
echo [INFO] Python modules installed successfully.
echo.

REM == Part 2: Equivalent of install_dependencies.sh ==
echo [INFO] Setting up dependencies (llama.cpp and unsloth_zoo)...
echo      This part attempts to follow the README's description for
echo      install_dependencies.sh. Manual steps might be needed.
echo.

REM --- llama.cpp ---
echo [INFO] Handling llama.cpp installation...
echo      The README states: "install specific versions of llama.cpp ... in this repo."
echo      This typically means cloning the repository and building it.
echo      This script does NOT automate the build process for llama.cpp as it can be complex
echo      and system-dependent (e.g., requiring C++ compilers like MSVC, CMake, etc.).
echo.
echo      ACTION REQUIRED FOR LLAMA.CPP:
echo      1. If the project expects a pre-built llama.cpp or specific source version,
echo         you'll need to obtain it and place it in this project directory as intended.
echo      2. If it needs to be cloned and built (common for llama.cpp):
echo         - git clone <URL_TO_LLAMA_CPP_REPO> llama.cpp
echo         - cd llama.cpp
echo         - git checkout <SPECIFIC_VERSION_TAG_OR_COMMIT_FROM_README_OR_ORIGINAL_SCRIPT>
echo         - Follow llama.cpp's build instructions for Windows (often involving CMake and a C++ compiler).
echo         - cd ..
echo      Consult the original install_dependencies.sh or project documentation if available
echo      for the exact version and source of llama.cpp.
echo.
echo      Press any key to acknowledge and continue with unsloth_zoo setup...
pause > nul
echo.

REM --- unsloth_zoo ---
echo [INFO] Handling unsloth_zoo installation and patching...
echo      The README states: "install specific versions of ... unsloth_zoo in this repo."
echo      and "my script will install this module in the current folder ... and patch it".
echo      This implies unsloth_zoo is a directory within your project, not just a pip package.
echo.

REM Check for Git (needed for cloning and patching)
git --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Git not found in PATH. Steps for unsloth_zoo cloning and patching will likely require manual execution using Git for Windows.
) else (
    echo [INFO] Git found.
)
echo.

REM Step 1: Get unsloth_zoo (assuming it's cloned into a directory named 'unsloth_zoo')
if exist unsloth_zoo (
    echo [INFO] Directory unsloth_zoo already exists. Assuming it's the correct version or you'll manage it manually.
) else (
    echo [INFO] Directory unsloth_zoo not found.
    echo      ACTION REQUIRED: You need to obtain the specific version of unsloth_zoo required
    echo      by this project and place it in a directory named 'unsloth_zoo' here.
    echo      This might involve:
    echo      - git clone <URL_TO_UNSLOTH_ZOO_REPO_OR_FORK> unsloth_zoo
    echo      - cd unsloth_zoo
    echo      - git checkout <SPECIFIC_VERSION_TAG_OR_COMMIT>
    echo      - cd ..
    echo      The original install_dependencies.sh would contain the exact source and version.
)
echo.

REM Step 2: Patch unsloth_zoo
echo [INFO] Attempting to patch unsloth_zoo...
echo      The README mentions patching unsloth_zoo to fix a bug. This script assumes
echo      a patch file (e.g., 'unsloth_zoo.patch') exists in the project root
echo      (c:\Sabareesh\Vector\Jupyter\) and is designed to be applied from within the 'unsloth_zoo' directory.
echo.
echo      ACTION REQUIRED (if not already done and unsloth_zoo directory exists):
echo      1. Ensure 'unsloth_zoo' directory contains the code to be patched.
echo      2. Identify the patch file name (e.g., unsloth_zoo.patch).
echo      3. If such a patch file exists and Git is available, you can try:
echo         cd unsloth_zoo
echo         git apply ../your_patch_file_name.patch
echo         cd ..
echo      Replace 'your_patch_file_name.patch' with the actual name.
echo      The original install_dependencies.sh would have the precise patching command.
echo.

echo ################################################################################
echo # Setup Script Finished                                                        #
echo ################################################################################
echo.
echo [SUMMARY & NEXT STEPS]
echo - Python virtual environment (.venv) should be set up. Requirements from
echo   requirements.txt have been attempted. Check for any errors above.
echo - For llama.cpp: Manual acquisition/build and placement is likely required.
echo - For unsloth_zoo: Ensure the correct version is in the 'unsloth_zoo' directory
echo   and that it has been patched as per the project's requirements.
echo.
echo - Please carefully review all output above for errors or warnings.
echo - CRITICAL: Consult the c:\Sabareesh\Vector\Jupyter\README.md for:
echo   - Detailed explanations of each component.
echo   - Dataset preparation (`conversations.csv`).
echo   - How to run training (`training-*.ipynb` via `run-training.sh` - you'll need to adapt notebook execution for Windows).
echo   - How to run inference (`inference-*.ipynb`).
echo - The README mentions specific Python (3.12.3) and NVIDIA driver versions.
echo   Ensure your Windows environment aligns as closely as possible.
echo - The notebooks expect to use the patched 'unsloth_zoo' from the local folder.
echo.
echo To activate the virtual environment manually in Command Prompt for subsequent work:
echo   %cd%\.venv\Scripts\activate.bat
echo In PowerShell:
echo   %cd%\.venv\Scripts\Activate.ps1
echo.

endlocal