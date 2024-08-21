import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

@st.cache_data
def fetch_fpl_data(num_games=38, min_minutes=0):
    st.write("Fetching FPL data...")
    
    # Fetch player and team data
    players_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    players_response = requests.get(players_url)
    players_data = players_response.json().get('elements', [])
    teams_data = players_response.json().get('teams', [])
    
    if not players_data or not teams_data:
        st.error("Failed to fetch player or team data.")
        return pd.DataFrame()

    # Convert to DataFrame
    df_players = pd.DataFrame(players_data)
    
    # Map team names and positions
    team_dict = {team['id']: team['name'] for team in teams_data}
    df_players['team_name'] = df_players['team'].map(team_dict)
    df_players['position'] = df_players['element_type'].map({
        1: 'Goalkeeper',
        2: 'Defender',
        3: 'Midfielder',
        4: 'Forward'
    })
    
    # Fetch fixture data
    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
    fixtures_response = requests.get(fixtures_url)
    fixtures_data = fixtures_response.json()
    
    if not fixtures_data:
        st.error("Failed to fetch fixture data.")
        return df_players

    # Convert to DataFrame
    fixture_data = pd.DataFrame(fixtures_data)
    fixture_data['home_team_name'] = fixture_data['team_h'].map(team_dict)
    fixture_data['away_team_name'] = fixture_data['team_a'].map(team_dict)

    # Calculate fixture difficulty ratings
    fdr_dict = {team: [] for team in team_dict.values()}
    for fixture in fixtures_data:
        home_team = team_dict[fixture['team_h']]
        away_team = team_dict[fixture['team_a']]
        if len(fdr_dict[home_team]) < num_games:
            fdr_dict[home_team].append(fixture['team_h_difficulty'])
        if len(fdr_dict[away_team]) < num_games:
            fdr_dict[away_team].append(fixture['team_a_difficulty'])
    
    average_fdr = {team: round(sum(fdrs) / len(fdrs), 2) for team, fdrs in fdr_dict.items()}
    df_players['average_fixture_difficulty'] = df_players['team_name'].map(average_fdr)
    
    # Prepare player data
    df_players['points_per_90'] = df_players['total_points'] / (df_players['minutes'] / 90)
    
    # Filter players by minutes
    df_players = df_players[df_players['minutes'] > min_minutes]
    
    # Check if data is available after filtering
    if df_players.empty:
        st.write("Data is insufficient after filtering.")
        return df_players
    
    # Feature Engineering and Scaling
    df_players['log_minutes'] = np.log1p(df_players['minutes'])
    features = [
        'log_minutes', 'goals_scored', 'assists', 'clean_sheets', 
        'goals_conceded', 'bonus', 'average_fixture_difficulty'
    ]
    target = 'points_per_90'
    
    if df_players[features].empty or df_players[target].empty:
        st.write("Features or target data is insufficient for model training.")
        return df_players
    
    X = df_players[features]
    y = df_players[target]

    # Feature Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature Selection
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = RFE(model, n_features_to_select=5)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Train the model using Random Forest
    model.fit(X_train, y_train)
    
    # Evaluate the model
    scores = cross_val_score(model, X_selected, y, cv=5)
    st.write(f"Model Cross-Validation Scores: {scores}")
    st.write(f"Average CV Score: {scores.mean()}")
    
    df_players['predicted_points_per_90'] = np.round(model.predict(X_selected), 2)
    
    # Add next fixtures for each player
    df_players['next_3_fixtures'] = df_players['team_name'].apply(lambda team: get_next_fixtures(team, fixture_data, num_games))

    # Add player price
    df_players['price'] = df_players['now_cost'] / 10.0  # Player prices are given in tenths
    
    # Add additional metrics
    df_players['selected_by_percent'] = df_players['selected_by_percent']  # Show the percentage of teams selecting the player
    df_players['transfers_out_event'] = df_players['transfers_out_event']  # Show transfers out during the current event
    
    return df_players

def get_next_fixtures(team_name, fixture_data, num_games=3):
    fixtures = fixture_data[
        (fixture_data['home_team_name'] == team_name) | (fixture_data['away_team_name'] == team_name)
    ]
    fixtures = fixtures[['event', 'home_team_name', 'away_team_name', 'team_h_difficulty', 'team_a_difficulty']]
    fixtures = fixtures.sort_values(by='event').head(num_games)
    return fixtures

def main():
    st.title('Fantasy Premier League Player Analysis')
    
    # Create a 2-column layout
    col1, col2 = st.columns([1, 2])  # Adjust the width ratio as needed

    with col1:
        # Number of games for fixture difficulty calculation
        num_games = st.slider('Number of Games for Fixture Difficulty Calculation', min_value=1, max_value=38, value=38, step=1)
        
        # Minimum minutes slider with a default value of 0
        min_minutes = st.slider('Minimum Minutes Played', min_value=0, max_value=1000, value=0, step=1)

    # Fetch player data based on the sliders
    df_players = fetch_fpl_data(num_games=num_games, min_minutes=min_minutes)

    if df_players.empty:
        st.write("No data available.")
        return

    # Get unique teams and positions for select boxes
    teams = df_players['team_name'].unique()
    positions = df_players['position'].unique()

    with col2:
        # Create select boxes and text input for filtering
        selected_team = st.selectbox('Select Team', ['All'] + list(teams))
        selected_position = st.selectbox('Select Position', ['All'] + list(positions))
        selected_player_name = st.text_input('Search by Player Name', '')
    
    # Filter data
    filtered_df = df_players
    if selected_team != 'All':
        filtered_df = filtered_df[filtered_df['team_name'] == selected_team]
    if selected_position != 'All':
        filtered_df = filtered_df[filtered_df['position'] == selected_position]
    if selected_player_name:
        filtered_df = filtered_df[filtered_df['first_name'].str.contains(selected_player_name, case=False) |
                                  filtered_df['second_name'].str.contains(selected_player_name, case=False)]
    
    filtered_df = filtered_df.sort_values(by='predicted_points_per_90', ascending=False)
    
    # Display filtered players
    if filtered_df.empty:
        st.write("No players match the selected criteria.")
    else:
        st.dataframe(filtered_df[[
            'first_name', 'second_name', 'team_name', 'position', 
            'predicted_points_per_90', 'expected_goals', 'expected_assists', 
            'goals_scored', 'assists', 'clean_sheets', 
            'average_fixture_difficulty', 'selected_by_percent', 'transfers_out_event', 'price'
        ]].rename(columns={
            'first_name': 'First Name',
            'second_name': 'Last Name',
            'team_name': 'Team',
            'position': 'Position',
            'predicted_points_per_90': 'Predicted Points per 90',
            'expected_goals': 'Expected Goals',
            'expected_assists': 'Expected Assists',
            'goals_scored': 'Goals Scored',
            'assists': 'Assists',
            'clean_sheets': 'Clean Sheets',
            'average_fixture_difficulty': 'Avg Fixture Difficulty',
            'selected_by_percent': 'Selected by (%)',
            'transfers_out_event': 'Transfers Out (Event)',
            'price': 'Price'
        }), use_container_width=True)

        # Show next 3 fixtures if checkbox is selected
        if st.checkbox('Show next 3 fixtures for selected players'):
            for player in filtered_df.itertuples():
                st.subheader(f"{player.first_name} {player.second_name} - {player.team}")
                next_fixtures_df = player.next_3_fixtures
                if not next_fixtures_df.empty:
                    st.dataframe(next_fixtures_df[[
                        'event', 'home_team_name', 'away_team_name', 
                        'team_h_difficulty', 'team_a_difficulty'
                    ]].rename(columns={
                        'home_team_name': 'Home Team',
                        'away_team_name': 'Away Team',
                        'team_h_difficulty': 'Home Team Difficulty',
                        'team_a_difficulty': 'Away Team Difficulty'
                    }))
                else:
                    st.write("No upcoming fixtures available.")

if __name__ == '__main__':
    main()