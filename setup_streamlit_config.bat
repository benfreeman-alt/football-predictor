@echo off
REM Setup Streamlit Configuration Files

echo Creating .streamlit directory...
mkdir "C:\Users\User\Desktop\prediction-markets\.streamlit" 2>nul

echo Creating config.toml...
(
echo [theme]
echo primaryColor = "#1f77b4"
echo backgroundColor = "#ffffff"
echo secondaryBackgroundColor = "#f0f2f6"
echo textColor = "#262730"
echo font = "sans serif"
echo.
echo [server]
echo headless = true
echo port = 8501
echo enableCORS = false
echo enableXsrfProtection = true
echo.
echo [browser]
echo gatherUsageStats = false
) > "C:\Users\User\Desktop\prediction-markets\.streamlit\config.toml"

echo Creating secrets.toml.template...
(
echo # Streamlit Cloud Secrets Configuration
echo # Add your API keys in Streamlit Cloud dashboard
echo.
echo # To set secrets in Streamlit Cloud:
echo # 1. Go to your app settings
echo # 2. Click "Secrets"
echo # 3. Add these values:
echo.
echo FOOTBALL_DATA_TOKEN = "your_token_here"
echo ODDS_API_KEY = "your_odds_api_key_here"
echo API_FOOTBALL_KEY = "your_api_football_key_here"
) > "C:\Users\User\Desktop\prediction-markets\.streamlit\secrets.toml.template"

echo.
echo ✓ Streamlit configuration files created!
echo.
echo Files created:
echo   - .streamlit\config.toml
echo   - .streamlit\secrets.toml.template
echo.
pause