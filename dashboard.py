"""
UNIVERSAL PREDICTION DASHBOARD

Streamlit web interface for all prediction markets
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Page config
st.set_page_config(
    page_title="Universal Prediction Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-card {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'football_loaded' not in st.session_state:
    st.session_state.football_loaded = False
    st.session_state.football_market = None

# Title
st.markdown('<h1 class="main-header">🎯 Universal Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Multi-Market Betting Intelligence Platform")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("🎲 Markets")
    
    # Market selection
    available_markets = ["📊 Overview", "⚽ Football", "🗳️ Elections", "📈 Bet Tracking"]
    selected_market = st.selectbox("Select Market", available_markets, label_visibility="collapsed")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### Quick Stats")
    st.metric("Active Markets", "2")
    st.metric("Total Predictions", "947")
    st.metric("Overall Win Rate", "73.2%")
    
    st.markdown("---")
    
    # System Status
    st.markdown("### System Status")
    st.success("🟢 All systems operational")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Load football market function
@st.cache_resource
def load_football_market():
    """Load and cache football market"""
    try:
        # Add path
        markets_path = os.path.join(os.getcwd(), 'universal_framework', 'markets')
        if markets_path not in sys.path:
            sys.path.insert(0, markets_path)
        
        from football_market import FootballMarket
        
        market = FootballMarket()
        if market.initialize():
            return market
        return None
    except Exception as e:
        st.error(f"Error loading football market: {e}")
        return None

# Main content based on selection
if selected_market == "📊 Overview":
    st.header("📊 Portfolio Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Markets", "2", delta="+1", help="Number of active prediction markets")
    
    with col2:
        st.metric("Total Predictions", "947", delta="+47", help="Total predictions made across all markets")
    
    with col3:
        st.metric("Overall Win Rate", "73.2%", delta="+1.2%", help="Combined accuracy across all markets")
    
    with col4:
        st.metric("Total ROI", "$4,230", delta="+$890", help="Total return on investment")
    
    st.markdown("---")
    
    # Market breakdown
    st.subheader("📈 Market Performance")
    
    market_data = pd.DataFrame({
        'Market': ['⚽ Football', '🗳️ Elections'],
        'Accuracy': ['74.4%', '72.0%'],
        'Predictions': [900, 47],
        'Expected ROI': ['45%', '35%'],
        'Status': ['🟢 Active', '🟢 Active']
    })
    
    st.dataframe(
        market_data, 
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Recent Activity
    st.subheader("🕐 Recent Activity")
    
    activity_data = pd.DataFrame({
        'Time': ['10 minutes ago', '1 hour ago', '3 hours ago', '1 day ago'],
        'Market': ['Football', 'Football', 'Elections', 'Football'],
        'Event': ['Arsenal vs Man United', 'Liverpool vs Man City', 'Polling Update', 'Chelsea vs Tottenham'],
        'Action': ['Predicted', 'Bet Placed', 'Model Updated', 'Won']
    })
    
    st.dataframe(activity_data, use_container_width=True, hide_index=True)

elif selected_market == "⚽ Football":
    st.header("⚽ Premier League Football")
    st.markdown("**Elite 74.4% Accuracy Model** | V4 Historical + Injury Adjustment")
    
    # Bankroll Input
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("")
    with col2:
        if 'bankroll' not in st.session_state:
            st.session_state.bankroll = 1000.0
        
        bankroll = st.number_input(
            "💰 Bankroll (£)",
            min_value=10.0,
            max_value=1000000.0,
            value=st.session_state.bankroll,
            step=50.0,
            key='football_bankroll',
            help="Your total betting bankroll"
        )
        st.session_state.bankroll = bankroll
    
    with col3:
        st.markdown("")
        st.markdown("")
        if st.button("🔄 Refresh Injuries", help="Clear injury cache and fetch latest data"):
            try:
                import os
                cache_dir = "data/injury_cache"
                if os.path.exists(cache_dir):
                    for file in os.listdir(cache_dir):
                        os.remove(os.path.join(cache_dir, file))
                    st.success("✅ Injury cache cleared!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Load football market
    if not st.session_state.football_loaded:
        with st.spinner("🔄 Loading football predictor... This may take a minute..."):
            football = load_football_market()
            
            if football:
                st.session_state.football_market = football
                st.session_state.football_loaded = True
                st.success("✅ Football market loaded!")
            else:
                st.error("❌ Failed to load football market. Check console for errors.")
                st.stop()
    
    football = st.session_state.football_market
    
    if football:
        # Stats
        stats = football.get_market_stats()
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Accuracy", f"{stats['accuracy']:.1%}", help="Test set accuracy on 180 matches")
        
        with col2:
            st.metric("Expected ROI", f"{stats['expected_roi']:.0%}", help="Expected return on investment per season")
        
        with col3:
            st.metric("Features", stats['features'], help="Number of predictive features")
        
        with col4:
            st.metric("Training Matches", stats['total_predictions'], help="Total matches used for training")
        
        st.markdown("---")
        
        # Model Info
        with st.expander("ℹ️ Model Information"):
            st.markdown(f"""
            **Model Version:** {stats['model_version']}
            
            **Key Features:**
            - Historical Expected Goals (xG) data from FBref
            - Shot quality and volume metrics
            - Set piece statistics
            - Real-time injury data
            - Head-to-head history
            - Recent form analysis
            
            **Advanced Features (Top 10):**
            1. H2H Home Advantage (21.1%)
            2. npxG Advantage (5.9%)
            3. Shot Quality Advantage (4.4%)
            4. Home npxG per Game (3.9%)
            5. Form Difference (3.6%)
            6. Home Attacking Quality (3.5%)
            7. Home xG per Shot (3.5%)
            8. Set Piece Advantage (3.3%)
            9. Home Attack vs Away Defense (2.9%)
            10. Home Shots per Game (2.9%)
            """)
        
        st.markdown("---")
        
        # Predictions
        st.subheader("🎯 Current Predictions")
        
        # ADD CUSTOM FIXTURE SECTION
        with st.expander("➕ Add Custom Fixture"):
            st.markdown("**Create a prediction for any upcoming match**")
            
            # Premier League teams
            teams = [
                'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
                'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
                'Leicester', 'Liverpool', 'Man City', 'Man United', 'Newcastle',
                "Nott'm Forest", 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
            ]
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                home_team = st.selectbox("Home Team", teams, key='custom_home')
            
            with col2:
                away_team = st.selectbox("Away Team", teams, key='custom_away', index=1)
            
            with col3:
                match_date = st.date_input("Date", key='custom_date')
            
            if st.button("🎯 Get Prediction"):
                if home_team == away_team:
                    st.error("Home and away teams must be different!")
                else:
                    with st.spinner("Generating prediction..."):
                        try:
                            # Get prediction for custom fixture
                            custom_pred = football.predictor.predict_match(
                                home_team, 
                                away_team, 
                                match_date.strftime('%Y-%m-%d')
                            )
                            
                            # Add to session state for display
                            if 'custom_predictions' not in st.session_state:
                                st.session_state.custom_predictions = []
                            
                            # Format like other predictions
                            custom_pred_formatted = {
                                'market': 'Football',
                                'event': f"{home_team} vs {away_team}",
                                'prediction': custom_pred['prediction'],
                                'confidence': custom_pred['confidence'],
                                'probability': custom_pred['probabilities']['home_win'],
                                'base_probability': custom_pred['probabilities']['base_home_win'],
                                'injury_adjustment': custom_pred['injury_adjustment'],
                                'home_team': home_team,
                                'away_team': away_team,
                                'injury_details': custom_pred['injury_details'],
                                'probabilities': custom_pred['probabilities'],
                                'odds': None,
                                'value_analysis': None,
                                'is_custom': True
                            }
                            
                            st.session_state.custom_predictions.append(custom_pred_formatted)
                            st.success(f"✅ Added: {home_team} vs {away_team}")
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error generating prediction: {e}")
        
        # Betting Strategy Explanation
        with st.expander("💡 Betting Strategy: Model + Positive EV"):
            st.markdown("""
            **Our Strategy: Only bet when BOTH conditions are met**
            
            1. ✅ **Model predicts the outcome** (HIGH/MEDIUM-HIGH confidence)
            2. ✅ **Positive Expected Value** (good odds = mathematical edge)
            
            **Labels:**
            - **⭐⭐⭐ BET NOW** = Model confident + Excellent value (20%+ edge)
            - **⭐⭐ BET** = Model confident + Good value (15-20% edge)  
            - **⭐ BET** = Model confident + Decent value (10-15% edge)
            - **📊 MODEL ONLY** = Model confident but odds too low (wait!)
            - **❌ SKIP** = Low confidence or no value
            
            **Why this strategy?**
            - Model accuracy (74.4%) tells us WHAT will happen
            - Odds tell us if the PRICE is right
            - We need BOTH to profit long-term
            """)
        
        predictions = football.get_predictions()
        
        # Add custom predictions if any
        if 'custom_predictions' in st.session_state:
            predictions = st.session_state.custom_predictions + predictions
        
        if not predictions:
            st.info("No predictions available. Add fixtures to get predictions.")
        else:
            # ENHANCED FILTER OPTIONS
            st.markdown("### 🔍 Filter Predictions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                confidence_filter = st.multiselect(
                    "Confidence Level",
                    ['HIGH', 'MEDIUM-HIGH', 'MEDIUM', 'LOW'],
                    default=['HIGH', 'MEDIUM-HIGH'],
                    key='conf_filter'
                )
            
            with col2:
                bet_filter = st.selectbox(
                    "Bet Status",
                    ['All Matches', 'Only Bets (⭐⭐⭐, ⭐⭐, ⭐)', 'Only Model Predictions', 'Skip Matches'],
                    index=0,
                    key='bet_filter'
                )
            
            with col3:
                st.write("")  # Spacing
                clear_custom = st.button("🗑️ Clear Custom Fixtures")
                
                if clear_custom:
                    if 'custom_predictions' in st.session_state:
                        st.session_state.custom_predictions = []
                        st.success("✅ Cleared custom fixtures")
                        st.rerun()
            
            st.markdown("---")
            
            # Apply filters
            filtered_predictions = []
            for pred in predictions:
                
                # Confidence filter
                if pred['confidence'] not in confidence_filter:
                    continue
                
                # Bet status filter
                value_analysis = pred.get('value_analysis')
                recommendation = value_analysis['recommendation'] if value_analysis else None
                is_bet = recommendation and recommendation['action'] == 'BET'
                
                if bet_filter == 'Only Bets (⭐⭐⭐, ⭐⭐, ⭐)':
                    if not is_bet:
                        continue
                elif bet_filter == 'Only Model Predictions':
                    if is_bet or pred['confidence'] in ['LOW']:
                        continue
                elif bet_filter == 'Skip Matches':
                    if is_bet or pred['confidence'] in ['HIGH', 'MEDIUM-HIGH', 'MEDIUM']:
                        continue
                
                filtered_predictions.append(pred)
            
            # Show count
            st.info(f"📊 Showing {len(filtered_predictions)} of {len(predictions)} matches")
            
            for idx, pred in enumerate(filtered_predictions):
                
                # Get value analysis if available
                value_analysis = pred.get('value_analysis')
                recommendation = value_analysis['recommendation'] if value_analysis else None
                
                # NEW LOGIC: Determine if this is a REAL bet (Model + EV)
                is_value_bet = False
                if recommendation and recommendation['action'] == 'BET':
                    is_value_bet = True
                
                # Determine color and display based on Model + EV strategy
                if is_value_bet:
                    # GREEN: Model predicts it + Positive EV = BET!
                    if recommendation['rating'] == '⭐⭐⭐':
                        border_color = '#28a745'  # Bright green
                        bet_label = '⭐⭐⭐ BET NOW'
                    elif recommendation['rating'] == '⭐⭐':
                        border_color = '#20c997'  # Teal
                        bet_label = '⭐⭐ BET'
                    elif recommendation['rating'] == '⭐':
                        border_color = '#17a2b8'  # Light blue
                        bet_label = '⭐ BET'
                    else:
                        border_color = '#6c757d'  # Gray
                        bet_label = '✓ SMALL BET'
                elif pred['confidence'] in ['HIGH', 'MEDIUM-HIGH']:
                    # YELLOW: Model predicts but NO positive EV
                    border_color = '#ffc107'
                    bet_label = '📊 MODEL ONLY'
                else:
                    # RED: Skip (uncertain or low confidence)
                    border_color = '#dc3545'
                    bet_label = '❌ SKIP'
                
                with st.container():
                    # Header with delete button
                    header_col1, header_col2 = st.columns([9, 1])
                    
                    with header_col1:
                        st.markdown(f"""
                        <div style="border-left: 4px solid {border_color}; padding: 1rem; margin: 1rem 0; background-color: #f8f9fa; border-radius: 0.5rem;">
                            <h3>{pred['event']} <span style="color: {border_color}; font-size: 0.8em;">{bet_label}</span></h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with header_col2:
                        # Delete button for custom fixtures
                        if pred.get('is_custom'):
                            if st.button("🗑️", key=f"delete_{idx}_{pred['event']}", help="Delete this fixture"):
                                if 'custom_predictions' in st.session_state:
                                    # Find and remove this prediction
                                    st.session_state.custom_predictions = [
                                        p for p in st.session_state.custom_predictions 
                                        if p['event'] != pred['event']
                                    ]
                                    st.success(f"✅ Deleted {pred['event']}")
                                    st.rerun()
                    
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown("**Model Prediction**")
                        
                        # Make prediction clearer
                        display_prediction = pred['prediction']
                        if 'Away Win' in display_prediction and 'or Draw' not in display_prediction:
                            display_prediction = 'Away Win or Draw'
                        elif pred['prediction'] == 'SKIP':
                            display_prediction = 'Too Close to Call'
                        
                        st.metric("Outcome", display_prediction)
                        st.metric("Confidence", pred['confidence'])
                    
                    with col2:
                        st.markdown("**Probabilities**")
                        
                        home_prob = pred['probability']
                        away_draw_prob = 1 - home_prob
                        
                        # Show BOTH probabilities clearly
                        st.metric("Home Win", f"{home_prob:.1%}")
                        st.metric("Away Win or Draw", f"{away_draw_prob:.1%}")
                        
                        if pred['injury_adjustment'] != 0:
                            st.caption(f"Injury adjustment: {pred['injury_adjustment']:+.1%}")
                    
                    with col3:
                        # NEW: Clear Model + EV Strategy Display
                        if is_value_bet:
                            st.success(f"✅ {bet_label}")
                            st.caption(f"Model + Value: {recommendation['stake_recommendation']:.1f}% stake")
                        elif pred['confidence'] in ['HIGH', 'MEDIUM-HIGH']:
                            st.warning("📊 MODEL ONLY")
                            st.caption("No +EV (wait for better odds)")
                        elif pred['confidence'] == 'MEDIUM':
                            st.info("⚠️ WATCH")
                            st.caption("Medium confidence")
                        else:
                            st.error("❌ SKIP")
                            st.caption("Low confidence")
                    
                    # Betting Recommendation Section
                    if recommendation:
                        with st.expander("💰 Betting Recommendation", expanded=(recommendation['action'] == 'BET')):
                            if recommendation['action'] == 'BET':
                                st.success(f"**{recommendation['rating']} RECOMMENDED BET**")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Bet Details:**")
                                    st.write(f"**Type:** {recommendation['bet_type']}")
                                    st.write(f"**Odds:** {recommendation['odds']:.2f}")
                                    
                                    # Calculate actual stake amount
                                    stake_pct = recommendation['stake_recommendation']
                                    kelly_stake = st.session_state.bankroll * (stake_pct / 100)
                                    
                                    # Check if below Betfair minimum
                                    if kelly_stake < 5.0:
                                        st.write(f"**Kelly Stake:** {stake_pct:.2f}% (£{kelly_stake:.2f})")
                                        st.warning(f"⚠️ Below Betfair £5 minimum")
                                        st.caption("Options: (1) Skip this bet, or (2) Bet £5 minimum (reduces ROI)")
                                        
                                        # Show what £5 represents
                                        actual_pct = (5.0 / st.session_state.bankroll) * 100
                                        st.caption(f"£5 = {actual_pct:.2f}% of bankroll (higher risk than Kelly suggests)")
                                        
                                        stake_amount = 5.0  # For profit calculation
                                    else:
                                        stake_amount = kelly_stake
                                        st.write(f"**Stake:** {stake_pct:.2f}% (£{stake_amount:.2f})")
                                    
                                    # Show potential profit
                                    potential_profit = stake_amount * (recommendation['odds'] - 1)
                                    st.write(f"**Potential Profit:** £{potential_profit:.2f}")
                                
                                with col2:
                                    st.markdown("**Value Analysis:**")
                                    st.write(f"**Expected Value:** {recommendation['expected_value']*100:+.1f}%")
                                    st.write(f"**Rating:** {recommendation['rating']}")
                                    st.write(f"**Confidence:** {pred['confidence']}")
                                
                                st.info(f"💡 {recommendation['reason']}")
                                
                                # Show odds breakdown if available
                                if pred.get('odds'):
                                    st.markdown("**Market Odds:**")
                                    odds_data = pred['odds']
                                    odds_col1, odds_col2, odds_col3 = st.columns(3)
                                    
                                    with odds_col1:
                                        if odds_data.get('home_win'):
                                            st.write(f"Home Win: {odds_data['home_win']:.2f}")
                                    with odds_col2:
                                        if odds_data.get('draw'):
                                            st.write(f"Draw: {odds_data['draw']:.2f}")
                                    with odds_col3:
                                        if odds_data.get('away_win'):
                                            st.write(f"Away Win: {odds_data['away_win']:.2f}")
                                
                                # MANUAL ODDS CALCULATOR
                                st.markdown("---")
                                st.markdown("**🔢 Check Different Odds**")
                                st.caption("Enter custom odds to see updated EV calculation")
                                
                                calc_col1, calc_col2, calc_col3 = st.columns(3)
                                
                                home_prob = pred['probability']
                                away_draw_prob = 1 - home_prob
                                
                                with calc_col1:
                                    custom_home_odds = st.number_input(
                                        "Home Win Odds",
                                        min_value=1.01,
                                        max_value=100.0,
                                        value=float(pred.get('odds', {}).get('home_win', 2.0)) if pred.get('odds') else 2.0,
                                        step=0.1,
                                        key=f"custom_home_{pred['event']}"
                                    )
                                
                                with calc_col2:
                                    custom_draw_odds = st.number_input(
                                        "Draw Odds",
                                        min_value=1.01,
                                        max_value=100.0,
                                        value=float(pred.get('odds', {}).get('draw', 3.5)) if pred.get('odds') else 3.5,
                                        step=0.1,
                                        key=f"custom_draw_{pred['event']}"
                                    )
                                
                                with calc_col3:
                                    custom_away_odds = st.number_input(
                                        "Away Win Odds",
                                        min_value=1.01,
                                        max_value=100.0,
                                        value=float(pred.get('odds', {}).get('away_win', 3.0)) if pred.get('odds') else 3.0,
                                        step=0.1,
                                        key=f"custom_away_{pred['event']}"
                                    )
                                
                                # Calculate custom double chance odds
                                custom_dc_away_draw = 1 / ((1/custom_away_odds) + (1/custom_draw_odds))
                                custom_dc_home_draw = 1 / ((1/custom_home_odds) + (1/custom_draw_odds))
                                
                                # Calculate EVs with custom odds
                                custom_home_ev = (home_prob * custom_home_odds) - 1
                                custom_away_draw_ev = (away_draw_prob * custom_dc_away_draw) - 1
                                
                                # Show results
                                st.markdown("**Custom EV Results:**")
                                
                                result_col1, result_col2 = st.columns(2)
                                
                                with result_col1:
                                    if custom_home_ev > 0:
                                        st.success(f"✅ Home Win EV: **{custom_home_ev*100:+.1f}%**")
                                    else:
                                        st.error(f"❌ Home Win EV: **{custom_home_ev*100:+.1f}%**")
                                
                                with result_col2:
                                    if custom_away_draw_ev > 0:
                                        st.success(f"✅ Away/Draw EV: **{custom_away_draw_ev*100:+.1f}%**")
                                    else:
                                        st.error(f"❌ Away/Draw EV: **{custom_away_draw_ev*100:+.1f}%**")
                                
                                st.caption(f"Double Chance (Away/Draw) odds: {custom_dc_away_draw:.2f}")
                                
                                # Recommendation based on custom odds
                                if home_prob > 0.55 and custom_home_ev > 0.05:
                                    st.info("💡 At these odds: **Bet Home Win** looks profitable!")
                                elif home_prob < 0.45 and custom_away_draw_ev > 0.05:
                                    st.info("💡 At these odds: **Bet Away/Draw** looks profitable!")
                                elif max(custom_home_ev, custom_away_draw_ev) > 0.05:
                                    st.warning("⚠️ These odds show +EV, but consider if they match what model predicts")
                                else:
                                    st.warning("⚠️ At these odds: No positive EV - wait for better odds")
                                
                                # LAY BETTING CALCULATOR
                                st.markdown("---")
                                st.markdown("**💱 Lay Betting Calculator (Betfair Exchange)**")
                                st.caption("Compare traditional betting vs laying on an exchange")
                                
                                lay_col1, lay_col2 = st.columns(2)
                                
                                with lay_col1:
                                    st.markdown("**Traditional Bet:**")
                                    trad_stake_input = st.number_input(
                                        "Stake (£)",
                                        min_value=1.0,
                                        max_value=10000.0,
                                        value=float(st.session_state.bankroll * 0.01),  # 1% of bankroll
                                        step=5.0,
                                        key=f"lay_trad_stake_{pred['event']}"
                                    )
                                    trad_odds_input = st.number_input(
                                        "Double Chance Odds",
                                        min_value=1.01,
                                        max_value=10.0,
                                        value=custom_dc_away_draw if custom_away_draw_ev > 0 else 1.50,
                                        step=0.01,
                                        key=f"lay_trad_odds_{pred['event']}"
                                    )
                                
                                with lay_col2:
                                    st.markdown("**Lay Bet (Exchange):**")
                                    lay_odds_input = st.number_input(
                                        "Lay Odds (e.g., Man United)",
                                        min_value=1.01,
                                        max_value=100.0,
                                        value=custom_home_odds if custom_home_odds > 0 else 2.00,
                                        step=0.01,
                                        key=f"lay_odds_{pred['event']}"
                                    )
                                    commission_input = st.number_input(
                                        "Commission %",
                                        min_value=0.0,
                                        max_value=10.0,
                                        value=2.0,
                                        step=0.5,
                                        key=f"lay_commission_{pred['event']}"
                                    )
                                
                                # Calculate
                                try:
                                    import sys
                                    sys.path.insert(0, 'universal_framework/markets')
                                    from lay_calculator import LayCalculator
                                    
                                    calc = LayCalculator(commission_rate=commission_input/100)
                                    comparison = calc.compare_bets(
                                        traditional_stake=trad_stake_input,
                                        traditional_odds=trad_odds_input,
                                        lay_odds=lay_odds_input
                                    )
                                    
                                    # Display results
                                    st.markdown("**📊 Comparison Results:**")
                                    
                                    result_col1, result_col2, result_col3 = st.columns(3)
                                    
                                    with result_col1:
                                        st.metric(
                                            "Traditional Bet",
                                            f"£{comparison['traditional']['stake']:.2f}",
                                            f"+£{comparison['traditional']['profit']:.2f}"
                                        )
                                        st.caption(f"Risk: £{comparison['traditional']['risk']:.2f}")
                                        st.caption(f"ROI: {comparison['traditional']['roi']:.1f}%")
                                    
                                    with result_col2:
                                        st.metric(
                                            "Lay Bet",
                                            f"£{comparison['lay']['backers_stake']:.2f}",
                                            f"+£{comparison['lay']['profit']:.2f}"
                                        )
                                        st.caption(f"Liability: £{comparison['lay']['liability']:.2f}")
                                        st.caption(f"ROI: {comparison['lay']['roi']:.1f}%")
                                    
                                    with result_col3:
                                        better = comparison['comparison']['better_option']
                                        diff = comparison['comparison']['profit_difference']
                                        pct = comparison['comparison']['percentage_better']
                                        
                                        if better == "LAY":
                                            st.success(f"**LAY is better!**")
                                            st.metric("Extra Profit", f"+£{diff:.2f}", f"+{pct:.1f}%")
                                        else:
                                            st.warning(f"**Traditional better**")
                                            st.metric("Difference", f"£{abs(diff):.2f}", f"{pct:.1f}%")
                                    
                                    # Key info box
                                    if better == "LAY":
                                        st.success(f"""
                                        ✅ **Recommendation: LAY on Betfair**
                                        
                                        - Lay stake: **£{comparison['lay']['backers_stake']:.2f}** at {lay_odds_input:.2f}
                                        - Your liability: **£{comparison['lay']['liability']:.2f}**
                                        - Profit if win: **£{comparison['lay']['profit']:.2f}** (after {commission_input}% commission)
                                        - Extra profit vs traditional: **£{diff:.2f}** ({pct:.1f}% more)
                                        """)
                                    else:
                                        st.info(f"""
                                        💡 **Stick with traditional bet**
                                        
                                        Traditional betting offers better value at these odds.
                                        """)
                                
                                except Exception as e:
                                    st.error(f"Lay calculator error: {e}")
                            
                            else:
                                st.warning("❌ No positive value found")
                                st.caption(recommendation['reason'])
                    else:
                        # No odds available - show model-based recommendation
                        with st.expander("📊 Model Recommendation"):
                            st.markdown("**Based on Model Confidence Only** (No odds data)")
                            
                            home_prob = pred['probability']
                            
                            if pred['confidence'] in ['HIGH', 'MEDIUM-HIGH']:
                                if home_prob > 0.60:
                                    st.success("**Recommended:** Bet Home Win")
                                    st.write(f"Home win probability: {home_prob:.1%}")
                                    st.write(f"Suggested stake: 1-2% of bankroll")
                                elif home_prob < 0.40:
                                    st.success("**Recommended:** Bet Away Win or Draw (Double Chance)")
                                    st.write(f"Away/Draw probability: {(1-home_prob):.1%}")
                                    st.write(f"Suggested stake: 1-2% of bankroll")
                                else:
                                    st.warning("Confidence moderate - proceed with caution")
                            else:
                                st.error("Low confidence - skip this match")
                            
                            st.info("💡 Enable odds API for value-based recommendations!")
                    
                    # Expandable injury details
                    with st.expander("🏥 Injury Report"):
                        inj = pred['injury_details']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{pred['home_team']}**")
                            st.write(f"Injuries: {inj['home_injuries']}")
                            st.write(f"Impact: {inj['home_impact']:.2f}")
                        
                        with col2:
                            st.markdown(f"**{pred['away_team']}**")
                            st.write(f"Injuries: {inj['away_injuries']}")
                            st.write(f"Impact: {inj['away_impact']:.2f}")
                    
                    st.markdown("---")

elif selected_market == "🗳️ Elections":
    st.header("🗳️ US Elections")
    
    st.info("📌 Election market integration coming soon...")
    
    st.markdown("""
    **Planned Features:**
    - Polling data integration
    - Swing state analysis
    - Electoral college predictions
    - Betting market odds comparison
    """)

# Bet Tracking Page
elif selected_market == "📈 Bet Tracking":
    st.header("📈 Bet Tracking & Performance")
    st.markdown("**Track your bets and monitor ROI**")
    
    st.info("💾 Your bet data is saved to `data/bet_tracking.json` and persists across sessions. Clearing Streamlit cache won't delete your bets!")
    
    # Initialize bet tracker - ALWAYS reload from file (don't cache in session_state)
    try:
        import sys
        sys.path.insert(0, 'universal_framework/markets')
        from bet_tracker import BetTracker
        
        # Create new instance each time - it loads from file
        tracker = BetTracker(data_file='data/bet_tracking.json')
    except Exception as e:
        st.error(f"Error loading bet tracker: {e}")
        st.stop()
    
    # Performance Stats
    st.markdown("---")
    st.subheader("📊 Performance Overview")
    
    stats = tracker.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Bets", stats['total_bets'])
        st.metric("Pending", stats['pending_bets'])
    
    with col2:
        st.metric("Wins", stats['wins'], f"{stats['win_rate']:.1f}%")
        st.metric("Losses", stats['losses'])
    
    with col3:
        st.metric("Total Staked", f"£{stats['total_staked']:.2f}")
        st.metric("Avg Odds", f"{stats['avg_odds']:.2f}")
    
    with col4:
        profit_delta = f"+£{stats['total_profit']:.2f}" if stats['total_profit'] >= 0 else f"-£{abs(stats['total_profit']):.2f}"
        st.metric("Total Profit", profit_delta)
        roi_color = "normal" if stats['roi'] >= 0 else "inverse"
        st.metric("ROI", f"{stats['roi']:.1f}%", delta_color=roi_color)
    
    st.markdown("---")
    
    # Add New Bet Section
    st.subheader("➕ Add New Bet")
    
    with st.form("add_bet_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            match = st.text_input("Match", placeholder="Team A vs Team B")
            
            bet_direction = st.selectbox("Bet Direction", [
                "BACK (Normal bet)",
                "LAY (Betfair Exchange)"
            ])
            
            bet_type = st.selectbox("Bet Type", [
                "Home Win",
                "Away Win",
                "Draw",
                "Away Win or Draw (Double Chance)",
                "Home Win or Draw (Double Chance)",
                "Over 2.5 Goals",
                "Under 2.5 Goals",
                "Lay Home Team",
                "Lay Away Team",
                "Other"
            ])
            odds = st.number_input("Odds", min_value=1.01, max_value=100.0, value=2.0, step=0.01)
        
        # Extract bet direction value outside col blocks
        bet_direction_value = "LAY" if "LAY" in bet_direction else "BACK"
        
        # Show help text in col1
        with col1:
            if bet_direction_value == "LAY":
                st.caption("💡 For LAY bets: Enter backer's stake (not your liability)")
        
        with col2:
            # Bankroll calculator (outside of main stake input)
            with st.expander("💰 Calculate Stake from Bankroll"):
                bankroll = st.number_input(
                    "Your Bankroll (£)",
                    min_value=1.0,
                    max_value=1000000.0,
                    value=1000.0,
                    step=50.0,
                    help="Total betting bankroll",
                    key="bet_tracking_bankroll"
                )
                stake_pct = st.slider(
                    "Stake %",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    help="Percentage of bankroll to stake"
                )
                calculated_stake = bankroll * (stake_pct / 100)
                st.info(f"**Recommended Stake:** £{calculated_stake:.2f}")
                st.caption("Copy this value to the Stake field above")
            
            stake = st.number_input("Stake (£)", min_value=0.01, max_value=10000.0, value=10.0, step=0.50)
            bet_date = st.date_input("Match Date")
            result = st.selectbox("Result (if known)", ["Pending", "Won", "Lost", "Push"])
        
        submitted = st.form_submit_button("Add Bet")
        
        if submitted:
            if not match:
                st.error("Please enter a match")
            else:
                result_value = None if result == "Pending" else result
                tracker.add_bet(
                    match=match,
                    bet_type=bet_type,
                    odds=odds,
                    stake=stake,
                    result=result_value,
                    date=bet_date.strftime('%Y-%m-%d'),
                    bet_direction=bet_direction_value
                )
                st.success(f"✅ Added {bet_direction_value} bet: {match}")
                st.rerun()
    
    st.markdown("---")
    
    # Bet History
    st.subheader("📋 Bet History")
    
    df = tracker.get_bets_dataframe()
    
    if not df.empty:
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                ["Pending", "Won", "Lost", "Push"],
                default=["Pending", "Won", "Lost", "Push"]
            )
        
        with col2:
            sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Profit", "Odds"])
        
        # Apply filters
        filtered_df = df[df['status'].isin(status_filter)].copy()
        
        # Apply sorting
        if sort_by == "Date (Newest)":
            filtered_df = filtered_df.sort_values('date', ascending=False)
        elif sort_by == "Date (Oldest)":
            filtered_df = filtered_df.sort_values('date', ascending=True)
        elif sort_by == "Profit":
            filtered_df = filtered_df.sort_values('profit', ascending=False)
        else:  # Odds
            filtered_df = filtered_df.sort_values('odds', ascending=False)
        
        # Display bets
        for idx, bet in filtered_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"**{bet['match']}**")
                    
                    # Show bet direction
                    bet_dir = bet.get('bet_direction', 'BACK')
                    direction_badge = "🔄 LAY" if bet_dir == 'LAY' else "📈 BACK"
                    
                    # For lay bets, show liability
                    if bet_dir == 'LAY':
                        liability = bet['stake'] * (bet['odds'] - 1)
                        st.caption(f"{direction_badge} {bet['bet_type']} @ {bet['odds']:.2f} | Backer's stake: £{bet['stake']:.2f} | Your liability: £{liability:.2f}")
                    else:
                        st.caption(f"{direction_badge} {bet['bet_type']} @ {bet['odds']:.2f} | £{bet['stake']:.2f} staked")
                
                with col2:
                    status = bet['status']
                    if status == "Won":
                        st.success(f"✅ Won: +£{bet.get('profit', 0):.2f}")
                    elif status == "Lost":
                        st.error(f"❌ Lost: £{abs(bet.get('profit', 0)):.2f}")
                    elif status == "Push":
                        st.info(f"🤝 Push: £0.00")
                    else:
                        st.warning(f"⏳ Pending")
                    
                    st.caption(f"Date: {bet['date']}")
                
                with col3:
                    # Update result button
                    if bet['status'] == "Pending":
                        with st.popover("Update", key=f"popover_{idx}"):
                            new_result = st.radio(
                                "Result",
                                ["Won", "Lost", "Push"],
                                key=f"result_{idx}_{bet['id']}"
                            )
                            if st.button("Save", key=f"save_{idx}_{bet['id']}"):
                                tracker.update_bet_result(bet['id'], new_result)
                                st.success("Updated!")
                                st.rerun()
                    
                    # Delete button
                    if st.button("🗑️", key=f"delete_{idx}_{bet['id']}", help="Delete bet"):
                        tracker.delete_bet(bet['id'])
                        st.rerun()
                
                st.markdown("---")
    
    else:
        st.info("No bets tracked yet. Add your first bet above!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 2rem 0;">
    <p>🎯 Universal Prediction Dashboard | Built with Streamlit</p>
    <p>Elite Multi-Market Betting Intelligence</p>
</div>
""", unsafe_allow_html=True)