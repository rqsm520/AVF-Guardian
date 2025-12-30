@echo off
echo ========================================================
echo   AVF Guardian - GitHub Auto Uploader
echo   Nephrology Research AI Assistant (Trae)
echo ========================================================
echo.
echo [STEP 1] Please go to https://github.com/new and create a repository named "AVF-Guardian".
echo          Note: Do NOT initialize with README, .gitignore, or License.
echo.
set /p REPO_URL="[STEP 2] Paste the HTTPS URL of your new repository here (e.g., https://github.com/rqsm514/AVF-Guardian.git): "
echo.
echo [STEP 3] Initializing and Packaging...
git init
git add .
git commit -m "Initial release for AJKD submission: AVF Guardian Model"

echo.
echo [STEP 4] Linking to Remote Repository...
git branch -M main
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo [STEP 5] Uploading to GitHub...
git push -u origin main

echo.
echo ========================================================
echo   Upload Complete! 
echo   Now go to https://share.streamlit.io/ to deploy.
echo   Repository: %REPO_URL%
echo ========================================================
pause