# ⚽ Football Prediction Dashboard

Elite 74.4% accuracy betting system with automated fixtures, live odds, and value betting calculations.

## 🎯 Features

- **74.4% Accurate Predictions** - V4 model with historical xG data
- **Automatic Fixtures** - Updates every 12 hours from football-data.org
- **Live Odds Integration** - Real-time odds comparison
- **Value Betting** - Model + Positive EV strategy
- **Bet Tracking** - ROI monitoring and performance stats
- **Mobile Access** - Responsive design for phone/tablet
- **Bankroll Management** - Automatic stake calculation

## 🚀 Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/football-predictor.git
cd football-predictor

# Install dependencies
pip install -r requirements.txt

# Set environment variables
set FOOTBALL_DATA_TOKEN=your_token
set ODDS_API_KEY=your_key

# Run dashboard
streamlit run dashboard.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy `dashboard.py`
5. Add secrets in app settings

## 🔑 API Keys Required

Get free API keys from:
- [football-data.org](https://www.football-data.org/client/register) - Fixtures
- [the-odds-api.com](https://the-odds-api.com) - Live odds (optional)

## 📊 Model Performance

- **Accuracy:** 74.4%
- **Training Data:** 3 seasons (2022-2025)
- **Features:** xG, form, injuries, H2H
- **Expected ROI:** 50-70% annually

## 🎓 Usage

1. Set your bankroll (£1,000 default)
2. Filter to "Only Bets"
3. See recommended bets with stakes
4. Place bets
5. Track performance in Bet Tracking

## 📱 Mobile Access

Access from anywhere via Streamlit Cloud URL:
`https://your-app.streamlit.app`

## 🛡️ Disclaimer

This tool is for educational and entertainment purposes. Bet responsibly.

## 📄 License

MIT License - Free to use and modify
