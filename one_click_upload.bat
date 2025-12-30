@echo off
echo ========================================================
echo   AVF Guardian - GitHub Auto Uploader
echo ========================================================
echo.
echo [STEP 1] Please go to https://github.com/new and create a repository named "AVF-Guardian".
echo.
set /p REPO_URL="[STEP 2] Paste the HTTPS URL of your new repository here (e.g., https://github.com/rqsm514/AVF-Guardian.git): "
echo.
echo [STEP 3] Uploading files...
git remote add origin %REPO_URL%
git branch -M main
git push -u origin main
echo.
echo ========================================================
echo   Upload Complete! 
echo   Now go to https://share.streamlit.io/ to deploy.
echo ========================================================
pause