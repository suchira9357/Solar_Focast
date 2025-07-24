import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import math
import random
import time
import sim_config as CFG

# Add SCATTER_PROBABILITY to sim_config if not already there
if not hasattr(CFG, 'SCATTER_PROBABILITY'):
    CFG.SCATTER_PROBABILITY = 0.0002

class TurbidityModel:
    """Model atmospheric turbidity variations with caching."""
    
    def __init__(self, location: Dict):
        self.location = location
        self.base_turbidity = self._get_base_turbidity()
        
        # Cache for seasonal factors (computed once per month)
        self._seasonal_cache = {}
        self._current_month = None
        
    def _get_base_turbidity(self) -> float:
        """Get base turbidity for location."""
        # Simplified - could use actual data
        if self.location['altitude'] > 1000:
            return 2.0  # Mountain
        elif abs(self.location['latitude']) > 40:
            return 2.5  # Temperate
        else:
            return 3.5  # Tropical
            
    def get_turbidity(self, timestamp: datetime, weather_state) -> float:
        """Calculate current turbidity with caching."""
        month = timestamp.month
        
        # Check if we need to recompute seasonal factor
        if month != self._current_month:
            self._current_month = month
            self._seasonal_cache[month] = 1 + 0.3 * np.sin((month - 6) * np.pi / 6)
        
        seasonal_factor = self._seasonal_cache[month]
        
        # Humidity effect
        humidity_factor = 1 + 0.01 * (weather_state.humidity - 50)
        
        # After rain effect
        rain_factor = 0.7 if weather_state.precipitation > 0 else 1.0
            
        turbidity = self.base_turbidity * seasonal_factor * humidity_factor * rain_factor
        
        return max(1.5, min(6.0, turbidity))

class CloudModel:
    """Advanced cloud modeling for weather system."""
    
    def __init__(self):
        self.cloud_database = self._load_cloud_properties()
        
    def _load_cloud_properties(self) -> Dict:
        """Load detailed cloud properties."""
        return {
            'cumulus': {
                'base_height': 500,  # meters
                'thickness': 1000,
                'albedo': 0.65,
                'emissivity': 0.95,
                'liquid_water_path': 50  # g/m²
            },
            'stratus': {
                'base_height': 200,
                'thickness': 500,
                'albedo': 0.60,
                'emissivity': 0.98,
                'liquid_water_path': 100
            },
            'cirrus': {
                'base_height': 8000,
                'thickness': 2000,
                'albedo': 0.30,
                'emissivity': 0.50,
                'liquid_water_path': 5
            },
            'cumulonimbus': {
                'base_height': 500,
                'thickness': 10000,
                'albedo': 0.90,
                'emissivity': 1.0,
                'liquid_water_path': 500
            }
        }
    
    def get_cloud_properties(self, cloud_type: str) -> Dict:
        """Get detailed properties for cloud type."""
        return self.cloud_database.get(cloud_type, self.cloud_database['cumulus'])

@dataclass
class WeatherState:
    """Current weather conditions."""
    timestamp: datetime
    temperature: float  # Celsius
    pressure: float  # hPa
    humidity: float  # %
    wind_speed: float  # m/s
    wind_direction: float  # degrees
    cloud_cover: float  # fraction (0-1)
    cloud_type: str
    visibility: float  # km
    precipitation: float  # mm/hr
    solar_radiation: Dict  # DNI, DHI, GHI measurements

class OptimizedWeatherSystem:
    """
    Optimized weather system with vectorized operations and caching.
    """
    
    def __init__(self, location: Dict = None, seed=0):
        """
        Initialize weather system for specific location.
        
        Args:
            location: Dict with 'latitude', 'longitude', 'altitude', 'timezone'
        """
        # Default location (Sri Lanka)
        if location is None:
            location = {
                'latitude': 6.9271,
                'longitude': 79.8612,
                'altitude': 10.0,
                'timezone': 'Asia/Colombo'
            }
            
        self.location = location
        self.current_state = None
        self.forecast = []
        
        # Weather patterns
        self.diurnal_patterns = self._initialize_diurnal_patterns()
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        
        # Atmospheric parameters
        self.turbidity_model = TurbidityModel(location)
        self.cloud_model = CloudModel()
        
        # Import EnhancedWindField here to avoid circular imports
        from enhanced_wind_field import EnhancedWindField
        
        # Initialize wind field for cloud movement
        self.wind = EnhancedWindField()
        
        # Initialize cloud parcels list - limit size for performance
        self.parcels = []
        self.max_parcels = getattr(CFG, 'MAX_PARCELS', 12)
        self.sim_time = 0.0
        self.time_since_last_spawn = 0.0
        self.max_gap_sec = 3.0
        
        # Caching for expensive computations
        self._daily_cache = {}
        self._current_day = None
        self._diurnal_cache = {}
        self._season_cache = {}
        self._current_season = None
        
        # Pre-computed arrays for vectorized operations
        self._hour_array = np.arange(24, dtype=np.float32)
        self._pi_array = np.pi
        
        # Pattern state - simplified tracking
        self.pattern_intensity = 0.5
        self.pattern_direction = 0.0
        self.current_formation_probability = getattr(CFG, 'SPAWN_PROBABILITY', 0.04)
        self.humidity = 60.0
        self.temperature = 28.0
        
        # Pre-compute precipitation rates for faster lookup
        self._precip_rates = {
            'clear': 0, 'cirrus': 0, 'altocumulus': 0,
            'cumulus': 0.1, 'stratocumulus': 0.5,
            'nimbostratus': 2.0, 'cumulonimbus': 10.0
        }
        
        # Initialize random generator for reproducibility
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
    def _initialize_diurnal_patterns(self) -> Dict:
        """Initialize typical daily weather patterns."""
        return {
            'temperature': {
                'min_hour': 6,
                'max_hour': 15,
                'amplitude': 8  # Daily temperature range
            },
            'humidity': {
                'min_hour': 15,
                'max_hour': 6,
                'amplitude': 20
            },
            'wind_speed': {
                'min_hour': 6,
                'max_hour': 15,
                'amplitude': 3
            },
            'cloud_formation': {
                'morning_fog': {'start': 5, 'end': 8, 'probability': 0.3},
                'afternoon_cumulus': {'start': 11, 'end': 17, 'probability': 0.4},
                'evening_clear': {'start': 18, 'end': 22, 'probability': 0.7}
            }
        }
    
    def _initialize_seasonal_patterns(self) -> Dict:
        """Initialize seasonal weather variations."""
        return {
            'summer': {
                'temp_range': (25, 35),
                'humidity_range': (50, 80),
                'cloud_types': ['cumulus', 'cumulonimbus'],
                'turbidity': 4.0
            },
            'monsoon': {
                'temp_range': (22, 30),
                'humidity_range': (70, 95),
                'cloud_types': ['nimbostratus', 'cumulonimbus'],
                'turbidity': 5.0
            },
            'winter': {
                'temp_range': (18, 28),
                'humidity_range': (40, 70),
                'cloud_types': ['cirrus', 'altocumulus'],
                'turbidity': 2.5
            }
        }
    
    def _get_cached_daily_patterns(self, day_of_year: int) -> Dict:
        """Get or compute daily patterns for the given day."""
        if day_of_year in self._daily_cache:
            return self._daily_cache[day_of_year]
        
        # Compute diurnal patterns for this day (vectorized)
        hours = self._hour_array
        
        # Temperature pattern
        temp_pattern = self.diurnal_patterns['temperature']
        temp_phase = ((hours - temp_pattern['min_hour']) / 
                     (temp_pattern['max_hour'] - temp_pattern['min_hour'])) * 2 * self._pi_array
        temp_factors = 0.5 + 0.5 * np.sin(temp_phase - self._pi_array/2)
        
        # Humidity pattern (inverse of temperature)
        humid_pattern = self.diurnal_patterns['humidity']
        humid_phase = ((hours - humid_pattern['min_hour']) / 
                      (humid_pattern['max_hour'] - humid_pattern['min_hour'])) * 2 * self._pi_array
        humid_factors = 0.5 + 0.5 * np.sin(humid_phase - self._pi_array/2)
        
        # Wind pattern
        wind_pattern = self.diurnal_patterns['wind_speed']
        wind_phase = ((hours - wind_pattern['min_hour']) / 
                     (wind_pattern['max_hour'] - wind_pattern['min_hour'])) * 2 * self._pi_array
        wind_factors = 0.5 + 0.5 * np.sin(wind_phase - self._pi_array/2)
        
        patterns = {
            'temp_factors': temp_factors,
            'humid_factors': humid_factors,
            'wind_factors': wind_factors
        }
        
        # Cache the result
        self._daily_cache[day_of_year] = patterns
        
        # Keep only last 2 days in cache to save memory
        if len(self._daily_cache) > 2:
            oldest_day = min(self._daily_cache.keys())
            del self._daily_cache[oldest_day]
        
        return patterns
    
    def update_weather_pattern(self):
        """Update global weather pattern with optimized calculations."""
        # Simplified pattern update - use cached values when possible
        pattern_cycle = (self.sim_time / 3600) % 24
        
        # Use fast trigonometric approximation for real-time updates
        self.pattern_intensity = 0.5 + 0.5 * math.sin(pattern_cycle * math.pi / 12)
        
        # Update wind directions less frequently
        if int(self.sim_time) % 60 == 0:  # Update every minute
            pattern_angle = (self.sim_time / 7200) % 360
            self.pattern_direction = pattern_angle
        
        # Update temperature and humidity using simplified model
        hour = (self.sim_time / 3600) % 24
        
        # Use cached diurnal patterns
        day_of_year = int(self.sim_time / 86400) % 365
        patterns = self._get_cached_daily_patterns(day_of_year)
        
        # Interpolate for current hour
        hour_idx = int(hour)
        hour_frac = hour - hour_idx
        next_hour_idx = (hour_idx + 1) % 24
        
        # Linear interpolation between cached values
        temp_factor = patterns['temp_factors'][hour_idx] * (1 - hour_frac) + \
                     patterns['temp_factors'][next_hour_idx] * hour_frac
        humid_factor = patterns['humid_factors'][hour_idx] * (1 - hour_frac) + \
                      patterns['humid_factors'][next_hour_idx] * hour_frac
        
        self.temperature = 28 + 5 * (temp_factor - 0.5) * 2
        self.humidity = 60 + 20 * (humid_factor - 0.5) * 2
        
        # Update cloud formation probability
        base_prob = getattr(CFG, 'SPAWN_PROBABILITY', 0.04)
        time_factor = 1.0 + 0.5 * self.pattern_intensity
        humidity_factor = 1.0 + (self.humidity - 60) / 100
        
        self.current_formation_probability = base_prob * time_factor * humidity_factor
    
    def update(self, timestamp: datetime) -> WeatherState:
        """Update weather state with optimized calculations."""
        # Get season with caching
        season = self._get_cached_season(timestamp)
        seasonal_params = self.seasonal_patterns[season]
        
        # Calculate base values using cached patterns
        hour = timestamp.hour + timestamp.minute / 60
        day_of_year = timestamp.timetuple().tm_yday
        patterns = self._get_cached_daily_patterns(day_of_year)
        
        # Interpolate cached values for current hour
        hour_idx = int(hour)
        hour_frac = hour - hour_idx
        next_hour_idx = (hour_idx + 1) % 24
        
        # Temperature
        temp_min, temp_max = seasonal_params['temp_range']
        temp_factor = patterns['temp_factors'][hour_idx] * (1 - hour_frac) + \
                     patterns['temp_factors'][next_hour_idx] * hour_frac
        temp = temp_min + (temp_max - temp_min) * temp_factor
        
        # Humidity (inverse of temperature)
        humidity_min, humidity_max = seasonal_params['humidity_range']
        humid_factor = patterns['humid_factors'][hour_idx] * (1 - hour_frac) + \
                      patterns['humid_factors'][next_hour_idx] * hour_frac
        humidity = humidity_max + (humidity_min - humidity_max) * humid_factor
        
        # Wind
        wind_base = 2.0
        wind_amplitude = self.diurnal_patterns['wind_speed']['amplitude']
        wind_factor = patterns['wind_factors'][hour_idx] * (1 - hour_frac) + \
                     patterns['wind_factors'][next_hour_idx] * hour_frac
        wind_speed = wind_base + wind_amplitude * wind_factor
        
        # Wind direction (simplified calculation)
        wind_direction = (225 + np.random.normal(0, 20)) % 360
        
        # Pressure (small variations)
        pressure = 1013 + np.random.normal(0, 5)
        
        # Cloud cover and type (optimized determination)
        cloud_data = self._determine_clouds_fast(hour, season)
        
        # Visibility (vectorized calculation)
        visibility = self._calculate_visibility_fast(humidity, cloud_data['cloud_cover'])
        
        # Precipitation (lookup table)
        precipitation = self._calculate_precipitation_fast(cloud_data['cloud_type'], humidity)
        
        # Solar radiation (simplified)
        solar_radiation = self._calculate_solar_radiation_fast(
            timestamp, cloud_data['cloud_cover'], seasonal_params['turbidity']
        )
        
        # Create weather state
        self.current_state = WeatherState(
            timestamp=timestamp,
            temperature=temp,
            pressure=pressure,
            humidity=humidity,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            cloud_cover=cloud_data['cloud_cover'],
            cloud_type=cloud_data['cloud_type'],
            visibility=visibility,
            precipitation=precipitation,
            solar_radiation=solar_radiation
        )
        
        return self.current_state
    
    def _get_cached_season(self, timestamp: datetime) -> str:
        """Get season with caching."""
        month = timestamp.month
        
        if month != self._current_season:
            self._current_season = month
            if month in [6, 7, 8, 9]:
                season = 'monsoon'
            elif month in [3, 4, 5]:
                season = 'summer'
            else:
                season = 'winter'
            self._season_cache[month] = season
        
        return self._season_cache[month]
    
    def _determine_clouds_fast(self, hour: float, season: str) -> Dict:
        """Fast cloud determination using lookup tables."""
        cloud_patterns = self.diurnal_patterns['cloud_formation']
        seasonal_clouds = self.seasonal_patterns[season]['cloud_types']
        
        # Fast lookup for cloud formation periods
        cloud_cover = 0.0
        cloud_type = 'clear'
        
        # Use boolean operations for fast period checking
        is_morning_fog = 5 <= hour <= 8 and np.random.random() < 0.3
        is_afternoon_cumulus = 11 <= hour <= 17 and np.random.random() < 0.4
        is_evening_clear = 18 <= hour <= 22 and np.random.random() < 0.7
        
        if is_morning_fog:
            cloud_cover = np.random.uniform(0.6, 0.9)
            cloud_type = 'stratus'
        elif is_afternoon_cumulus:
            cloud_cover = np.random.uniform(0.3, 0.7)
            cloud_type = seasonal_clouds[np.random.randint(len(seasonal_clouds))]
        elif is_evening_clear:
            cloud_cover = np.random.uniform(0, 0.3)
            cloud_type = 'clear'
                        
        return {'cloud_cover': cloud_cover, 'cloud_type': cloud_type}
    
    def _calculate_visibility_fast(self, humidity: float, cloud_cover: float) -> float:
        """Fast visibility calculation using vectorized operations."""
        # Vectorized calculation
        base_vis = 50.0
        humidity_factor = np.exp(-0.02 * (humidity - 40))
        cloud_factor = 1 - 0.5 * cloud_cover
        visibility = base_vis * humidity_factor * cloud_factor
        
        return max(1.0, visibility)
    
    def _calculate_precipitation_fast(self, cloud_type: str, humidity: float) -> float:
        """Fast precipitation calculation using lookup table."""
        base_rate = self._precip_rates.get(cloud_type, 0)
        
        # Apply humidity threshold efficiently
        if base_rate > 0:
            humidity_thresholds = {
                'cumulus': 80, 'stratocumulus': 85, 
                'cumulonimbus': 75, 'nimbostratus': 0
            }
            threshold = humidity_thresholds.get(cloud_type, 100)
            
            if humidity > threshold:
                base_rate *= np.random.uniform(0.5, 1.5)
            else:
                base_rate = 0
                
        return base_rate
    
    def _calculate_solar_radiation_fast(self, timestamp: datetime, cloud_cover: float,
                                       turbidity: float) -> Dict:
        """Fast solar radiation calculation."""
        hour = timestamp.hour + timestamp.minute / 60
        
        if hour < 6 or hour > 18:
            # Return zeros for night time
            return {
                'dni': 0, 'dhi': 0, 'ghi': 0, 
                'clear_sky_dni': 0, 'clear_sky_ghi': 0,
                'elevation': 0, 'azimuth': 0
            }
        
        # Vectorized solar elevation calculation
        solar_elevation = 70 * np.sin(np.pi * (hour - 6) / 12)
        
        # Vectorized irradiance calculations
        elevation_factor = solar_elevation / 90
        clear_sky_dni = 1000 * elevation_factor
        clear_sky_dhi = 100 * elevation_factor
        clear_sky_ghi = clear_sky_dni * np.sin(np.radians(solar_elevation)) + clear_sky_dhi
        
        # Apply cloud effects
        cloud_factor = 1 - cloud_cover * 0.75
        diffuse_factor = 1 + cloud_cover * 0.2
        ghi_factor = 1 - cloud_cover * 0.5
        
        return {
            'dni': clear_sky_dni * cloud_factor,
            'dhi': clear_sky_dhi * diffuse_factor,
            'ghi': clear_sky_ghi * ghi_factor,
            'clear_sky_dni': clear_sky_dni,
            'clear_sky_ghi': clear_sky_ghi,
            'elevation': solar_elevation,
            'azimuth': 180 * (hour - 6) / 12
        }
    
    def _spawn(self, t):
        """Optimized cloud spawning."""
        # Import CloudParcel here to avoid circular imports
        try:
            from cloud_simulation import EnhancedCloudParcel
        except ImportError:
            print("Warning: EnhancedCloudParcel not available, cloud spawning disabled")
            return
        
        # Fast wind direction lookup
        try:
            _, hdg = self.wind.sample(CFG.DOMAIN_SIZE_M/2, CFG.DOMAIN_SIZE_M/2, 1000)
        except:
            hdg = 90  # Default East direction
            
        d = CFG.DOMAIN_SIZE_M
        
        # Vectorized spawn position calculation
        upwind_angle = (hdg + 180) % 360
        edge_distance = 0.1 * d
        center_x, center_y = d / 2, d / 2
        
        # Fast edge determination using lookup
        if 45 <= upwind_angle < 135:  # North edge
            spawn_x = center_x + random.uniform(-0.4, 0.4) * d
            spawn_y = -edge_distance
        elif 135 <= upwind_angle < 225:  # East edge
            spawn_x = d + edge_distance
            spawn_y = center_y + random.uniform(-0.4, 0.4) * d
        elif 225 <= upwind_angle < 315:  # South edge
            spawn_x = center_x + random.uniform(-0.4, 0.4) * d
            spawn_y = d + edge_distance
        else:  # West edge
            spawn_x = -edge_distance
            spawn_y = center_y + random.uniform(-0.4, 0.4) * d
        
        # Fast cloud type selection using numpy
        if self.humidity > 80:
            types = ["cirrus", "cumulus", "cumulonimbus"]
            weights = [0.1, 0.4, 0.5]
        elif self.humidity > 60:
            types = ["cirrus", "cumulus", "cumulonimbus"]
            weights = [0.3, 0.6, 0.1]
        else:
            types = ["cirrus", "cumulus", "cumulonimbus"]
            weights = [0.7, 0.2, 0.1]
            
        ctype = np.random.choice(types, p=weights)
        
        # Create new cloud parcel
        new_cloud = EnhancedCloudParcel(spawn_x, spawn_y, self.wind, ctype)
        self.parcels.append(new_cloud)
    
    def step(self, t=None, dt=None, t_s=None):
        """Optimized weather system step with minimal allocations."""
        # Update simulation time
        if t_s is not None:
            self.sim_time = t_s
        elif t is not None:
            self.sim_time = t
        else:
            self.sim_time += 1
        
        # Use physics timestep if dt not provided
        if dt is None:
            dt = getattr(CFG, 'PHYSICS_TIMESTEP', 1/60.0)
        
        # Update global weather pattern
        self.update_weather_pattern()
        
        # Update wind field
        if hasattr(self.wind, 'step'):
            self.wind.step(self.sim_time)
        
        # Optimized parcel updates - process in batches
        if self.parcels:
            # Process all parcels in vectorized operations where possible
            expired_indices = []
            for i, parcel in enumerate(self.parcels):
                if hasattr(parcel, 'step'):
                    if parcel.step(dt, self.wind, self.sim_time):
                        expired_indices.append(i)
            
            # Remove expired parcels efficiently
            if expired_indices:
                # Remove in reverse order to preserve indices
                for i in reversed(expired_indices):
                    self.parcels.pop(i)
        
        # Handle cloud scattering with minimal allocations
        if self.parcels:
            new_parcels = []
            for p in self.parcels:
                if getattr(p, "flag_for_split", False):
                    # Create fragments efficiently
                    n = random.randint(2, 3)
                    
                    for _ in range(n):
                        try:
                            from cloud_simulation import EnhancedCloudParcel
                            child = p.__class__(
                                p.x + random.uniform(-500, 500),
                                p.y + random.uniform(-500, 500),
                                p.wind, p.type)
                            child.vx, child.vy = p.vx, p.vy
                            child.r = p.r * random.uniform(0.8, 0.9)
                            child.age = p.growth_frames
                            new_parcels.append(child)
                        except ImportError:
                            break
                    
                    p.flag_for_split = False
                    p.split_fading = 60
                    new_parcels.append(p)
                else:
                    new_parcels.append(p)
            
            self.parcels = new_parcels
        
        # Optimized spawning logic
        self.time_since_last_spawn += dt
        
        # Check spawning conditions efficiently
        should_spawn = False
        
        if hasattr(CFG, 'SINGLE_CLOUD_MODE') and CFG.SINGLE_CLOUD_MODE:
            should_spawn = len(self.parcels) == 0
        else:
            max_parcels = self.max_parcels
            need_forced = (len(self.parcels) == 0 and 
                          self.time_since_last_spawn > self.max_gap_sec)
            
            should_spawn = (len(self.parcels) < max_parcels and 
                           (random.random() < self.current_formation_probability or 
                            need_forced))
        
        if should_spawn:
            self._spawn(self.sim_time)
            self.time_since_last_spawn = 0.0
    
    def get_avg_trajectory(self):
        """Optimized trajectory calculation using vectorized operations."""
        if not self.parcels:
            return None, None, 0
        
        # Vectorize velocity calculations
        velocities = np.array([(p.vx, p.vy) for p in self.parcels 
                              if hasattr(p, 'vx') and hasattr(p, 'vy')])
        
        if len(velocities) == 0:
            return None, None, 0
        
        # Vectorized speed and direction calculations
        speeds = np.linalg.norm(velocities, axis=1)
        directions = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        # Convert to unit vectors and average
        unit_vectors = np.column_stack([np.cos(directions), np.sin(directions)])
        mean_vector = np.mean(unit_vectors, axis=0)
        
        avg_speed = np.mean(speeds)
        avg_direction = np.degrees(np.arctan2(mean_vector[1], mean_vector[0])) % 360
        confidence = np.linalg.norm(mean_vector)
        
        # Convert m/s to km/h
        avg_speed_kmh = avg_speed * 3.6
        return avg_speed_kmh, avg_direction, confidence
    
    def current_cloud_cover_pct(self):
        """Optimized cloud cover calculation."""
        if not self.parcels:
            return 0.0
        
        # Vectorized area calculation
        radii = np.array([p.r for p in self.parcels if hasattr(p, 'r')])
        if len(radii) == 0:
            return 0.0
        
        total_area = np.sum(np.pi * radii**2)
        domain_area = getattr(CFG, 'AREA_SIZE_KM', 50)**2
        
        cover = min(100, (total_area / domain_area) * 100 * 5)
        return cover
    
    def get_forecast(self, hours_ahead: int = 24) -> List[WeatherState]:
        """Generate weather forecast efficiently."""
        forecast = []
        current = datetime.now()
        
        # Pre-allocate and vectorize where possible
        timestamps = [current + timedelta(hours=h) for h in range(hours_ahead)]
        
        for timestamp in timestamps:
            weather = self.update(timestamp)
            forecast.append(weather)
            
        return forecast
    
    def get_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate historical weather data efficiently."""
        # Calculate number of data points
        total_minutes = int((end_date - start_date).total_seconds() / 300)  # 5-minute intervals
        
        # Pre-allocate arrays for better performance
        timestamps = []
        data_arrays = {
            'temperature': np.zeros(total_minutes),
            'humidity': np.zeros(total_minutes),
            'wind_speed': np.zeros(total_minutes),
            'wind_direction': np.zeros(total_minutes),
            'cloud_cover': np.zeros(total_minutes),
            'dni': np.zeros(total_minutes),
            'dhi': np.zeros(total_minutes),
            'ghi': np.zeros(total_minutes)
        }
        
        current = start_date
        idx = 0
        
        while current <= end_date and idx < total_minutes:
            weather = self.update(current)
            
            timestamps.append(weather.timestamp)
            data_arrays['temperature'][idx] = weather.temperature
            data_arrays['humidity'][idx] = weather.humidity
            data_arrays['wind_speed'][idx] = weather.wind_speed
            data_arrays['wind_direction'][idx] = weather.wind_direction
            data_arrays['cloud_cover'][idx] = weather.cloud_cover
            data_arrays['dni'][idx] = weather.solar_radiation['dni']
            data_arrays['dhi'][idx] = weather.solar_radiation['dhi']
            data_arrays['ghi'][idx] = weather.solar_radiation['ghi']
            
            current += timedelta(minutes=5)
            idx += 1
        
        # Create DataFrame efficiently
        data_dict = {'timestamp': timestamps}
        data_dict.update({k: v[:idx] for k, v in data_arrays.items()})
        
        return pd.DataFrame(data_dict)


# Alias for backward compatibility
WeatherSystem = OptimizedWeatherSystem