"""
WEATHER DATA MODULE

Gets current weather conditions for match locations
Free OpenWeather API
"""

import requests
from datetime import datetime

class WeatherTracker:
    """Track weather conditions for matches"""
    
    # Stadium locations (latitude, longitude)
    STADIUM_LOCATIONS = {
        'Arsenal': (51.5549, -0.1084),  # Emirates Stadium
        'Man City': (53.4831, -2.2004),  # Etihad
        'Liverpool': (53.4308, -2.9608),  # Anfield
        'Man United': (53.4631, -2.2913),  # Old Trafford
        'Chelsea': (51.4817, -0.1910),  # Stamford Bridge
        'Tottenham': (51.6042, -0.0664),  # Tottenham Hotspur Stadium
        'Newcastle': (54.9756, -1.6217),  # St James' Park
        'Brighton': (50.8609, -0.0830),  # Amex Stadium
        'Aston Villa': (52.5092, -1.8848),  # Villa Park
        'West Ham': (51.5383, -0.0164),  # London Stadium
        'Crystal Palace': (51.3983, -0.0853),  # Selhurst Park
        'Fulham': (51.4748, -0.2215),  # Craven Cottage
        'Brentford': (51.4907, -0.2889),  # Brentford Community Stadium
        'Wolves': (52.5901, -2.1306),  # Molineux
        'Bournemouth': (50.7352, -1.8382),  # Vitality Stadium
        'Everton': (53.4387, -2.9660),  # Goodison Park
        'Leicester': (52.6203, -1.1420),  # King Power Stadium
        'Southampton': (50.9058, -1.3910),  # St Mary's
        'Ipswich': (52.0550, 1.1448),  # Portman Road
        "Nott'm Forest": (52.9400, -1.1327),  # City Ground
    }
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.weather_data = {}
    
    def get_weather(self, team_name):
        """
        Get current weather for team's stadium
        
        Returns: Dict with weather conditions
        """
        
        if self.api_key is None or self.api_key == 'YOUR_API_KEY':
            # Return neutral weather if no API
            return {
                'temperature': 15,  # Neutral temp
                'rain': 0,
                'wind_speed': 5,
                'condition': 'clear'
            }
        
        if team_name not in self.STADIUM_LOCATIONS:
            return self.get_default_weather()
        
        lat, lon = self.STADIUM_LOCATIONS[team_name]
        
        url = f"https://api.openweathermap.org/data/2.5/weather"
        
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            weather = {
                'temperature': data['main']['temp'],
                'rain': data.get('rain', {}).get('1h', 0),  # Rain in last hour
                'wind_speed': data['wind']['speed'],
                'condition': data['weather'][0]['main'].lower()
            }
            
            return weather
        
        except:
            return self.get_default_weather()
    
    def get_default_weather(self):
        """Default weather (average UK conditions)"""
        return {
            'temperature': 12,
            'rain': 0,
            'wind_speed': 5,
            'condition': 'clear'
        }
    
    def get_weather_impact(self, team_name):
        """
        Calculate weather impact score
        
        Heavy rain/wind negatively affects technical teams
        """
        
        weather = self.get_weather(team_name)
        
        impact = 0
        
        # Heavy rain affects passing game
        if weather['rain'] > 5:
            impact += 2
        elif weather['rain'] > 2:
            impact += 1
        
        # Strong wind affects game
        if weather['wind_speed'] > 15:
            impact += 2
        elif weather['wind_speed'] > 10:
            impact += 1
        
        # Extreme cold
        if weather['temperature'] < 2:
            impact += 1
        
        return impact