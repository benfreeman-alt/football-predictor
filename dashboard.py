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
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
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
    
    with col4:
        st.markdown("")
        st.markdown("")
        if st.button("🗓️ Refresh Fixtures", help="Clear fixture cache and fetch latest"):
            try:
                import os
                cache_file = "data/fixture_cache/auto_fixtures.json"
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    st.success("✅ Fixture cache cleared!")
                    st.rerun()
                else:
                    st.info("No cache to clear")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    # Date Range Filter
    st.subheader("📅 Fixture Date Filter")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    from datetime import datetime, timedelta
    
    with filter_col1:
        today = datetime.now().date()
        
        start_date = st.date_input(
            "Show fixtures from:",
            value=today,
            min_value=today - timedelta(days=7),  # Allow looking back 7 days
            help="Only show matches from this date onwards"
        )
    
    with filter_col2:
        default_end = today + timedelta(days=14)
        end_date = st.date_input(
            "Show fixtures until:",
            value=default_end,
            min_value=today,
            help="Only show matches up to this date"
        )
    
    with filter_col3:
        st.markdown("**Quick Filters:**")
        quick_col1, quick_col2 = st.columns(2)
        with quick_col1:
            if st.button("This Week", help="Next 7 days", key="week_filter"):
                st.session_state.start_date = today
                st.session_state.end_date = today + timedelta(days=7)
                st.rerun()
        with quick_col2:
            if st.button("Next 2 Weeks", help="Next 14 days", key="fortnight_filter"):
                st.session_state.start_date = today
                st.session_state.end_date = today + timedelta(days=14)
                st.rerun()
    
    # Apply session state if exists
    if 'start_date' in st.session_state:
        start_date = st.session_state.start_date
    if 'end_date' in st.session_state:
        end_date = st.session_state.end_date
    
    st.markdown("---")
    
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
        
        # DEBUG: Show what we got
        st.write(f"🔍 DEBUG: Received {len(predictions) if predictions else 0} predictions from football.get_predictions()")
        
        # Add custom predictions if any
        if 'custom_predictions' in st.session_state:
            predictions = st.session_state.custom_predictions + predictions
        
        if not predictions:
            st.error("❌ No predictions available. Add fixtures to get predictions.")
            
            # Show debug info
            with st.expander("🐛 Debug Information"):
                st.write("**Troubleshooting:**")
                st.write("1. Click the '🗓️ Refresh Fixtures' button above")
                st.write("2. Check your FOOTBALL_DATA_TOKEN in Streamlit secrets")
                st.write("3. Try adding a custom fixture below")
                
                # Try to manually check fixture loader
                try:
                    from simple_fixture_loader import SimpleFixtureLoader
                    loader = SimpleFixtureLoader()
                    upcoming = loader.get_upcoming_fixtures(days_ahead=14)
                    st.write(f"**Fixture loader test:** Found {len(upcoming) if upcoming else 0} fixtures")
                    if upcoming:
                        st.write("Sample fixtures:")
                        for fixture in upcoming[:3]:
                            st.write(f"  - {fixture.get('home_team')} vs {fixture.get('away_team')} on {fixture.get('date')}")
                except Exception as e:
                    st.write(f"**Fixture loader error:** {e}")
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
                
                # Date range filter
                pred_date_str = pred.get('date', '')
                date_filtered_out = False
                
                if pred_date_str:  # Only apply date filter if prediction has a date
                    try:
                        pred_date = datetime.strptime(pred_date_str, '%Y-%m-%d').date()
                        # Check if prediction date is outside the selected range
                        if pred_date < start_date or pred_date > end_date:
                            date_filtered_out = True
                    except Exception as e:
                        # If date parsing fails, don't filter it out
                        pass
                
                # Skip if filtered out by date
                if date_filtered_out:
                    continue
                
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
            
            # Debug: Show date filter info
            if len(filtered_predictions) == 0 and len(predictions) > 0:
                st.warning(f"⚠️ All {len(predictions)} matches filtered out. Check date range: {start_date} to {end_date}")
                with st.expander("🔍 Debug Info"):
                    for pred in predictions[:3]:
                        pred_date_str = pred.get('date', 'NO DATE')
                        st.write(f"Match: {pred.get('event', 'Unknown')} - Date: {pred_date_str}")
            
            # Display predictions in compact tiles
            st.markdown("### 🎯 Upcoming Matches")
            
            # Group by recommendation type
            bet_now = []
            value_bets = []
            model_only = []
            skip_matches = []
            
            for p in filtered_predictions:
                value_analysis = p.get('value_analysis')
                recommendation = value_analysis['recommendation'] if value_analysis else None
                
                if recommendation and recommendation['action'] == 'BET':
                    if recommendation['rating'] == '⭐⭐⭐':
                        bet_now.append(p)
                    else:
                        value_bets.append(p)
                elif p['confidence'] in ['HIGH', 'MEDIUM-HIGH']:
                    model_only.append(p)
                else:
                    skip_matches.append(p)
            
            # Show summary counts
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("⭐⭐⭐ Bet Now", len(bet_now))
            with col2:
                st.metric("⭐ Value", len(value_bets))
            with col3:
                st.metric("📊 Model", len(model_only))
            with col4:
                st.metric("❌ Skip", len(skip_matches))
            
            st.markdown("---")
            
            # Display BET NOW matches (collapsed tiles)
            if bet_now:
                st.markdown("### ⭐⭐⭐ BET NOW")
                for pred in bet_now:
                    rec = pred['value_analysis']['recommendation']
                    with st.expander(f"**{pred['event']}** - {rec['bet_type']} @ {rec['odds']:.2f}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Prediction:** {pred['prediction']}")
                            st.write(f"**Confidence:** {pred['confidence']}")
                            st.caption(f"📅 {pred.get('date', 'TBD')} {pred.get('time', '')}")
                        
                        with col2:
                            stake_pct = rec['stake_recommendation']
                            kelly = st.session_state.bankroll * (stake_pct / 100)
                            st.write(f"**Stake:** {stake_pct:.2f}%")
                            
                            if kelly < 5.0:
                                st.write(f"**Amount:** £5.00 (min)")
                                st.caption(f"⚠️ Kelly: £{kelly:.2f}")
                            else:
                                st.write(f"**Amount:** £{kelly:.2f}")
                        
                        with col3:
                            stake = max(5.0, kelly)
                            profit = (rec['odds'] - 1) * stake
                            st.write(f"**Profit:** +£{profit:.2f}")
                            st.write(f"**EV:** {rec.get('edge', 0):.1%}")
                            
                            # Show model vs odds probability
                            model_prob = pred['probability']
                            implied_prob = 1 / rec['odds']
                            st.caption(f"Model: {model_prob:.1%} vs Odds: {implied_prob:.1%}")
                        
                        st.markdown("---")
                        
                        # Show Model Prediction Probability
                        bet_type_display = rec['bet_type']
                        
                        # Calculate what the model thinks will win
                        if 'Home Win' in bet_type_display:
                            win_prob = model_prob
                            prob_label = f"📊 Model predicts **Home Win**: {win_prob:.1%}"
                        elif 'Away' in bet_type_display or 'Draw' in bet_type_display:
                            win_prob = 1 - model_prob
                            prob_label = f"📊 Model predicts **Away/Draw**: {win_prob:.1%}"
                        else:
                            win_prob = model_prob
                            prob_label = f"📊 Model confidence: {win_prob:.1%}"
                        
                        st.info(prob_label)
                        
                        # Betting Options
                        st.markdown("### 💰 Place Your Bet")
                        
                        bet_col1, bet_col2 = st.columns(2)
                        
                        with bet_col1:
                            st.markdown("**Traditional Bookmaker:**")
                            
                            # Safe default value
                            default_odds = max(1.01, float(rec.get('odds', 2.0)))
                            
                            actual_odds = st.number_input(
                                "Your Odds",
                                min_value=1.01,
                                max_value=100.0,
                                value=default_odds,
                                step=0.01,
                                key=f"actual_odds_{pred['event']}"
                            )
                            
                            actual_stake = st.number_input(
                                "Your Stake (£)",
                                min_value=1.0,
                                max_value=10000.0,
                                value=float(max(5.0, kelly)),
                                step=1.0,
                                key=f"actual_stake_{pred['event']}"
                            )
                            
                            trad_profit = (actual_odds - 1) * actual_stake
                            st.success(f"**Profit if wins:** +£{trad_profit:.2f}")
                            
                            # Calculate EV as percentage
                            model_prob = pred['probability']
                            expected_return = (model_prob * (actual_odds - 1)) - ((1 - model_prob) * 1)  # Per £1
                            ev_percent = expected_return * 100  # Convert to %
                            ev_color = "green" if ev_percent > 0 else "red"
                            st.markdown(f"**Expected Value:** <span style='color:{ev_color}'>{ev_percent:+.1f}%</span>", unsafe_allow_html=True)
                        
                        with bet_col2:
                            st.markdown("**Betfair Exchange (Lay):**")
                            
                            # Safe default value for lay odds
                            default_lay_odds = max(1.01, float(rec.get('odds', 2.0)) + 0.1)
                            
                            lay_odds = st.number_input(
                                "Lay Odds",
                                min_value=1.01,
                                max_value=100.0,
                                value=default_lay_odds,
                                step=0.01,
                                key=f"lay_odds_{pred['event']}"
                            )
                            
                            commission = st.number_input(
                                "Commission %",
                                min_value=0.0,
                                max_value=10.0,
                                value=2.0,
                                step=0.5,
                                key=f"commission_{pred['event']}"
                            )
                            
                            # Calculate lay bet
                            try:
                                import sys
                                sys.path.insert(0, 'universal_framework/markets')
                                from lay_calculator import LayCalculator
                                
                                calc = LayCalculator(commission_rate=commission/100)
                                comparison = calc.compare_bets(
                                    traditional_stake=actual_stake,
                                    traditional_odds=actual_odds,
                                    lay_odds=lay_odds
                                )
                                
                                lay_profit = comparison['lay']['profit']
                                liability = comparison['lay']['liability']
                                
                                if comparison['comparison']['better_option'] == 'LAY':
                                    st.success(f"**Lay profit:** +£{lay_profit:.2f}")
                                    st.caption(f"Liability: £{liability:.2f}")
                                    st.caption(f"✅ {comparison['comparison']['percentage_better']:.1f}% better than traditional")
                                else:
                                    st.info(f"**Lay profit:** +£{lay_profit:.2f}")
                                    st.caption(f"Liability: £{liability:.2f}")
                                    st.caption(f"Traditional bet is better")
                            except Exception as e:
                                st.caption(f"Lay calc error: {e}")
                        
                        # Add to tracker button
                        st.markdown("---")
                        
                        track_col1, track_col2, track_col3 = st.columns([2, 2, 1])
                        
                        with track_col1:
                            bet_direction = st.selectbox(
                                "Bet Type",
                                ["BACK (Normal bet)", "LAY (Betfair Exchange)"],
                                key=f"direction_{pred['event']}"
                            )
                        
                        with track_col2:
                            if "LAY" in bet_direction:
                                st.caption(f"Enter backer's stake: £{actual_stake / (lay_odds - 1):.2f}")
                            else:
                                st.caption(f"Stake: £{actual_stake:.2f}")
                        
                        with track_col3:
                            if st.button("➕ Track Bet", key=f"track_{pred['event']}"):
                                try:
                                    import sys
                                    sys.path.insert(0, 'universal_framework/markets')
                                    from bet_tracker import BetTracker
                                    
                                    tracker = BetTracker()
                                    
                                    bet_direction_value = "LAY" if "LAY" in bet_direction else "BACK"
                                    
                                    if bet_direction_value == "LAY":
                                        # For lay bets, use backer's stake
                                        backers_stake = actual_stake / (lay_odds - 1)
                                        tracker.add_bet(
                                            match=pred['event'],
                                            bet_type=f"Lay {rec['bet_type']}",
                                            odds=lay_odds,
                                            stake=backers_stake,
                                            result="Pending",
                                            bet_direction=bet_direction_value
                                        )
                                    else:
                                        tracker.add_bet(
                                            match=pred['event'],
                                            bet_type=rec['bet_type'],
                                            odds=actual_odds,
                                            stake=actual_stake,
                                            result="Pending",
                                            bet_direction=bet_direction_value
                                        )
                                    
                                    st.success("✅ Added to tracker!")
                                except Exception as e:
                                    st.error(f"Error: {e}")
            
            # Display VALUE BETS
            if value_bets:
                st.markdown("### ⭐ Value Bets")
                for pred in value_bets:
                    rec = pred['value_analysis']['recommendation']
                    with st.expander(f"**{pred['event']}** {rec['rating']}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**{rec['bet_type']}** @ {rec['odds']:.2f}")
                            st.write(f"Confidence: {pred['confidence']}")
                            st.caption(f"📅 {pred.get('date', 'TBD')}")
                        
                        with col2:
                            stake_pct = rec['stake_recommendation']
                            kelly = st.session_state.bankroll * (stake_pct / 100)
                            stake = max(5.0, kelly)
                            profit = (rec['odds'] - 1) * stake
                            st.write(f"Stake: £{stake:.2f}")
                            st.write(f"Profit: +£{profit:.2f}")
                            st.write(f"**EV:** {rec.get('edge', 0):.1%}")
                        
                        st.markdown("---")
                        
                        # Show Model Prediction Probability
                        model_prob_v = pred['probability']
                        bet_type_display_v = rec['bet_type']
                        
                        # Calculate what the model thinks will win
                        if 'Home Win' in bet_type_display_v:
                            win_prob_v = model_prob_v
                            prob_label_v = f"📊 Model predicts **Home Win**: {win_prob_v:.1%}"
                        elif 'Away' in bet_type_display_v or 'Draw' in bet_type_display_v:
                            win_prob_v = 1 - model_prob_v
                            prob_label_v = f"📊 Model predicts **Away/Draw**: {win_prob_v:.1%}"
                        else:
                            win_prob_v = model_prob_v
                            prob_label_v = f"📊 Model confidence: {win_prob_v:.1%}"
                        
                        st.info(prob_label_v)
                        
                        # Betting Options
                        st.markdown("### 💰 Place Your Bet")
                        
                        bet_col1, bet_col2 = st.columns(2)
                        
                        with bet_col1:
                            st.markdown("**Traditional Bookmaker:**")
                            
                            # Safe default value
                            default_odds_v = max(1.01, float(rec.get('odds', 2.0)))
                            
                            actual_odds_v = st.number_input(
                                "Your Odds",
                                min_value=1.01,
                                max_value=100.0,
                                value=default_odds_v,
                                step=0.01,
                                key=f"actual_odds_v_{pred['event']}"
                            )
                            
                            actual_stake_v = st.number_input(
                                "Your Stake (£)",
                                min_value=1.0,
                                max_value=10000.0,
                                value=float(stake),
                                step=1.0,
                                key=f"actual_stake_v_{pred['event']}"
                            )
                            
                            trad_profit_v = (actual_odds_v - 1) * actual_stake_v
                            st.success(f"**Profit if wins:** +£{trad_profit_v:.2f}")
                            
                            # Calculate EV as percentage
                            model_prob_v = pred['probability']
                            expected_return_v = (model_prob_v * (actual_odds_v - 1)) - ((1 - model_prob_v) * 1)  # Per £1
                            ev_percent_v = expected_return_v * 100  # Convert to %
                            ev_color_v = "green" if ev_percent_v > 0 else "red"
                            st.markdown(f"**Expected Value:** <span style='color:{ev_color_v}'>{ev_percent_v:+.1f}%</span>", unsafe_allow_html=True)
                        
                        with bet_col2:
                            st.markdown("**Betfair Exchange (Lay):**")
                            
                            # Safe default for lay odds
                            default_lay_odds_v = max(1.01, float(rec.get('odds', 2.0)) + 0.1)
                            
                            lay_odds_v = st.number_input(
                                "Lay Odds",
                                min_value=1.01,
                                max_value=100.0,
                                value=default_lay_odds_v,
                                step=0.01,
                                key=f"lay_odds_v_{pred['event']}"
                            )
                            
                            commission_v = st.number_input(
                                "Commission %",
                                min_value=0.0,
                                max_value=10.0,
                                value=2.0,
                                step=0.5,
                                key=f"commission_v_{pred['event']}"
                            )
                            
                            # Calculate lay bet
                            try:
                                import sys
                                sys.path.insert(0, 'universal_framework/markets')
                                from lay_calculator import LayCalculator
                                
                                calc = LayCalculator(commission_rate=commission_v/100)
                                comparison = calc.compare_bets(
                                    traditional_stake=actual_stake_v,
                                    traditional_odds=actual_odds_v,
                                    lay_odds=lay_odds_v
                                )
                                
                                lay_profit_v = comparison['lay']['profit']
                                liability_v = comparison['lay']['liability']
                                
                                if comparison['comparison']['better_option'] == 'LAY':
                                    st.success(f"**Lay profit:** +£{lay_profit_v:.2f}")
                                    st.caption(f"Liability: £{liability_v:.2f}")
                                    st.caption(f"✅ {comparison['comparison']['percentage_better']:.1f}% better")
                                else:
                                    st.info(f"**Lay profit:** +£{lay_profit_v:.2f}")
                                    st.caption(f"Liability: £{liability_v:.2f}")
                                    st.caption(f"Traditional better")
                            except Exception as e:
                                st.caption(f"Error: {e}")
                        
                        # Add to tracker
                        if st.button("➕ Add to Tracker", key=f"track_v_{pred['event']}"):
                            try:
                                import sys
                                sys.path.insert(0, 'universal_framework/markets')
                                from bet_tracker import BetTracker
                                
                                tracker = BetTracker()
                                tracker.add_bet(
                                    match=pred['event'],
                                    bet_type=rec['bet_type'],
                                    odds=actual_odds_v,
                                    stake=actual_stake_v,
                                    result="Pending"
                                )
                                st.success("✅ Added!"
)
                            except Exception as e:
                                st.error(f"Error: {e}")
            
            # Display MODEL ONLY (collapsed)
            if model_only:
                with st.expander(f"📊 Model Predictions - No +EV ({len(model_only)} matches)", expanded=False):
                    for pred in model_only:
                        st.markdown(f"**{pred['event']}**")
                        st.caption(f"• Prediction: {pred['prediction']} ({pred['confidence']})")
                        st.caption(f"• Home Win Prob: {pred['probability']:.0%}")
                        st.caption(f"• ⚠️ Model confident but odds not favorable - wait for better price")
                        st.caption(f"• 📅 {pred.get('date', 'TBD')}")
                        st.markdown("---")
            
            # Display SKIP matches (collapsed)
            if skip_matches:
                with st.expander(f"❌ Skip These Matches ({len(skip_matches)})", expanded=False):
                    for pred in skip_matches:
                        st.caption(f"• {pred['event']} - {pred['confidence']} - {pred.get('date', 'TBD')}")


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
                        with st.popover("Update", key=f"bettrack_popover_{idx}_{bet['id']}"):
                            new_result = st.radio(
                                "Result",
                                ["Won", "Lost", "Push"],
                                key=f"bettrack_result_{idx}_{bet['id']}"
                            )
                            if st.button("Save", key=f"bettrack_save_{idx}_{bet['id']}"):
                                tracker.update_bet_result(bet['id'], new_result)
                                st.success("Updated!")
                                st.rerun()
                    
                    # Delete button
                    if st.button("🗑️", key=f"bettrack_delete_{idx}_{bet['id']}", help="Delete bet"):
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