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
import os
import sys
from panel_layout import PANELS, build_panel_cells, panel_df

# Add the renderer path to the system path
sys.path.append('C:/Users/Suchira_Garusinghe/Desktop/Simulation/simulation8.5.0/pygame_rendereres')

# Import OpenGL rendering modules
try:
    from pygame_rendereres.panel_renderer import initialize_gl_panel_renderer, gl_draw_solar_panels
    from pygame_rendereres.ui_renderer import initialize_gl_ui_renderer, gl_create_info_panel
    from pygame_rendereres.cloud_renderer import initialize_gl_for_pygame, render_gl_clouds, km_to_screen_coords
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: OpenGL renderers not available. Using Pygame renderers only.")

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
    
    def _spawn(self, t):
        """Optimized cloud spawning."""
        # Import UltraOptimizedCloudParcel for compatibility with renderer
        try:
            from cloud_simulation import UltraOptimizedCloudParcel
        except ImportError:
            print("Warning: UltraOptimizedCloudParcel not available, cloud spawning disabled")
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

        # Only use the correct constructor signature for UltraOptimizedCloudParcel
        try:
            new_cloud = UltraOptimizedCloudParcel(spawn_x, spawn_y, ctype)
        except Exception as e:
            print(f"[ERROR] UltraOptimizedCloudParcel could not be instantiated: {e}")
            return
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

# Alias for backward compatibility
WeatherSystem = OptimizedWeatherSystem

class SimulationController:
    """
    Main simulation controller that orchestrates the solar farm simulation.
    Optimized for performance with streamlined operations.
    """
    
    def __init__(self, start_time=11.0, duration_hours=None, timestep_minutes=5, 
                 real_time_output=False, debug_mode=False, panel_df=None):
        """
        Initialize simulation controller.
        
        Args:
            start_time: Starting hour of simulation
            duration_hours: Duration to run (None for infinite)
            timestep_minutes: Time step in minutes
            real_time_output: Enable real-time output
            debug_mode: Enable debug logging
            panel_df: Panel DataFrame
        """
        self.start_time = start_time
        self.duration_hours = duration_hours
        self.timestep_minutes = timestep_minutes
        self.real_time_output = real_time_output
        self.debug_mode = debug_mode
        self.current_hour = start_time
        self.panel_df = panel_df if panel_df is not None else panel_df
        self.CELL_SIZE_KM = 2.0
        self.panel_cells = build_panel_cells(self.CELL_SIZE_KM)
        
        # Initialize all subsystems
        self._initialize_systems()
        
        # Performance tracking
        self.frame_count = 0
        self.last_physics_time = time.time()
        self.accumulated_dt = 0.0
        
        # UI elements (lazy initialization)
        self.font = None
        self.large_font = None
        self.info_panel_bg = None
        
        # OpenGL rendering support
        self.use_gl_renderer = False
        self.gl_cloud_renderer = None
        self.gl_panel_renderer = None
        self.gl_ui_renderer = None
        self.screen_width = 1200
        self.screen_height = 900
    
    def _initialize_systems(self):
        """Initialize all simulation subsystems efficiently."""
        try:
            # Initialize weather system (optimized)
            self.weather_system = OptimizedWeatherSystem()
            print("Optimized weather system initialized")
            
            # Initialize shadow calculator
            from shadow_calculator import ShadowCalculator
            self.shadow_calculator = ShadowCalculator(
                domain_size=CFG.DOMAIN_SIZE_M,
                area_size_km=CFG.AREA_SIZE_KM
            )
            self.shadow_calculator.cloud_transmittance = CFG.CLOUD_TRANSMITTANCE
            self.shadow_calculator.shadow_fade_ms = CFG.SHADOW_FADE_MS
            self.shadow_calculator.spatial_cell_size = self.CELL_SIZE_KM
            
            print(f"Shadow calculator initialized (transmittance={self.shadow_calculator.cloud_transmittance})")
            print(f"Panel spatial index built with {len(self.panel_cells)} cells")
            
            # Initialize power simulator
            self._initialize_power_simulator()
            
        except Exception as e:
            print(f"Error during system initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _initialize_power_simulator(self):
        """Initialize power simulator with error handling."""
        try:
            from power_simulator import PowerSimulator
            self.power_simulator = PowerSimulator(
                panel_df=self.panel_df,
                latitude=6.9271,
                longitude=79.8612
            )
            print("Power simulator initialized")
            
            # Set default values if missing
            if not hasattr(self.power_simulator, 'sunrise_hour'):
                self.power_simulator.sunrise_hour = 6.5
            if not hasattr(self.power_simulator, 'sunset_hour'):
                self.power_simulator.sunset_hour = 18.5
            if not hasattr(self.power_simulator, 'calculate_solar_position'):
                self.power_simulator.calculate_solar_position = lambda hour: {
                    "elevation": 90 - abs(12-hour)*7, 
                    "azimuth": (180 + (hour-12)*15) % 360
                }
                
            print(f"Solar generation hours: {self.power_simulator.sunrise_hour:.1f}h to {self.power_simulator.sunset_hour:.1f}h")
            
        except Exception as e:
            print(f"Error initializing power simulator: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback power simulator
            from types import SimpleNamespace
            self.power_simulator = SimpleNamespace()
            self.power_simulator.calculate_power = lambda hour, coverage: {"total": 0.0}
            self.power_simulator.sunrise_hour = 6.5
            self.power_simulator.sunset_hour = 18.5
            self.power_simulator.calculate_solar_position = lambda hour: {"elevation": 90-abs(12-hour)*7, "azimuth": 180}
    
    def step(self):
        """
        Execute one simulation step with optimized physics timing.
        
        Returns:
            Dictionary with simulation results
        """
        self.frame_count += 1
        current_time = time.time()
        dt = current_time - self.last_physics_time
        self.last_physics_time = current_time
        self.accumulated_dt += dt
        
        # Cap maximum timestep to prevent large jumps
        max_dt = 0.1
        if self.accumulated_dt > max_dt:
            self.accumulated_dt = max_dt
        
        # Fixed timestep physics updates
        while self.accumulated_dt >= CFG.PHYSICS_TIMESTEP:
            # Update simulation time
            self.current_hour += self.timestep_minutes / 60.0 * (CFG.PHYSICS_TIMESTEP / (5.0 / 60.0))
            self.current_hour = self.current_hour % 24
            
            # Step weather system
            self.weather_system.step(self.frame_count)
            
            self.accumulated_dt -= CFG.PHYSICS_TIMESTEP
        
        # Create timestamp for current simulation time
        current_time_obj = datetime.now()
        hour = int(self.current_hour)
        minute = int((self.current_hour % 1) * 60)
        timestamp = current_time_obj.replace(hour=hour, minute=minute)
        
        # Get solar position
        solar_position = self.power_simulator.calculate_solar_position(self.current_hour)
        
        # Get cloud ellipses from weather system
        from cloud_simulation import collect_visible_ellipses
        cloud_ellipses = collect_visible_ellipses(self.weather_system.parcels)
        
        # Calculate shadow coverage
        ellipses_for_shadow = [e[:7] for e in cloud_ellipses]
        panel_coverage = self.shadow_calculator.calculate_panel_coverage(
            ellipses_for_shadow, self.panel_df, solar_position, self.panel_cells
        )
        
        # Calculate power output
        power_output = self.power_simulator.calculate_power(
            self.current_hour, panel_coverage
        )
        
        # Get trajectory information
        cloud_speed, cloud_direction, confidence = self.weather_system.get_avg_trajectory()
        
        # Debug output (throttled)
        if self.debug_mode and self.frame_count % 10 == 0:
            print(f"Hour: {self.current_hour:.2f}, Cloud count: {len(cloud_ellipses)}")
            print(f"Solar position: El={solar_position['elevation']:.1f}°, Az={solar_position['azimuth']:.1f}°")
            print(f"Cloud cover: {self.weather_system.current_cloud_cover_pct():.1f}%, Total power: {power_output.get('total', 0.0):.2f} kW")
            if cloud_speed is not None:
                print(f"Cloud movement: {cloud_speed:.1f} km/h, {cloud_direction:.0f}°")
        
        # Calculate interpolation alpha for smooth rendering
        alpha = self.accumulated_dt / CFG.PHYSICS_TIMESTEP if CFG.PHYSICS_TIMESTEP > 0 else 0.0
        alpha = min(0.99, max(0.01, alpha))  # Clamp for smoothness
        
        return {
            'time': timestamp,
            'cloud_ellipses': cloud_ellipses,
            'panel_coverage': panel_coverage,
            'power_output': power_output,
            'total_power': power_output.get('total', 0.0),
            'cloud_cover': self.weather_system.current_cloud_cover_pct(),
            'cloud_speed': cloud_speed,
            'cloud_direction': cloud_direction,
            'confidence': confidence,
            'solar_position': solar_position,
            'alpha': alpha  # Add interpolation factor
        }
    
    def toggle_gl_renderer(self):
        """Toggle between Pygame and OpenGL renderers"""
        if not OPENGL_AVAILABLE:
            print("OpenGL renderers not available. Cannot toggle.")
            return
            
        self.use_gl_renderer = not self.use_gl_renderer
        
        # Reset GL renderers if switching back to Pygame
        if not self.use_gl_renderer:
            self.gl_cloud_renderer = None
            self.gl_panel_renderer = None
            self.gl_ui_renderer = None
            
            # Recreate the Pygame window
            import pygame
            pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        
        print(f"Using {'OpenGL' if self.use_gl_renderer else 'Pygame'} renderer")
    
    def render_clouds(self, screen, cloud_ellipses, cloud_positions, x_range, y_range, width, height, alpha=0.0):
        """Render clouds using either Pygame or OpenGL with improved interpolation"""
        # Ensure alpha is properly clamped to improve smoothness
        alpha = max(0.01, min(0.99, alpha))  # Never use exactly 0 or 1 to ensure smoothness
        
        # Apply cubic easing for super-smooth motion
        eased_alpha = alpha * alpha * (3 - 2 * alpha)  # Cubic easing function
        
        if self.use_gl_renderer and OPENGL_AVAILABLE:
            # Initialize GL renderer if not already done
            if self.gl_cloud_renderer is None:
                self.gl_cloud_renderer = initialize_gl_for_pygame(screen)
                if self.gl_cloud_renderer is None:
                    # Fallback to Pygame if OpenGL initialization fails
                    self.use_gl_renderer = False
                    print("OpenGL initialization failed. Falling back to Pygame renderer.")
                else:
                    self.screen_width, self.screen_height = screen.get_size()
            
            # Render clouds with OpenGL if renderer is available
            if self.gl_cloud_renderer:
                # Create interpolated ellipses for smooth rendering
                interpolated_ellipses = []
                for i, ellipse_params in enumerate(cloud_ellipses):
                    if len(ellipse_params) >= 8:  # Has position data
                        # Get current position
                        cx, cy = ellipse_params[0], ellipse_params[1]
                        
                        # Find corresponding parcel for previous position
                        if i < len(self.weather_system.parcels):
                            parcel = self.weather_system.parcels[i]
                            if hasattr(parcel, 'prev_x') and hasattr(parcel, 'prev_y'):
                                # Interpolate position with eased alpha
                                interp_x = parcel.prev_x * (1 - eased_alpha) + cx * eased_alpha
                                interp_y = parcel.prev_y * (1 - eased_alpha) + cy * eased_alpha
                                
                                # Create interpolated ellipse
                                interp_ellipse = (interp_x, interp_y) + ellipse_params[2:]
                                interpolated_ellipses.append(interp_ellipse)
                            else:
                                interpolated_ellipses.append(ellipse_params)
                        else:
                            interpolated_ellipses.append(ellipse_params)
                    else:
                        interpolated_ellipses.append(ellipse_params)
                
                render_gl_clouds(screen, self.gl_cloud_renderer, interpolated_ellipses, x_range, y_range)
                return
        
        # Use the original Pygame renderer with improved interpolation
        from pygame_rendereres.cloud_renderer import draw_cloud_trail, create_cloud_surface
        
        # Draw cloud trail with interpolation
        interpolated_positions = []
        for pos in cloud_positions:
            interpolated_positions.append(pos)  # Trail positions don't need interpolation
        
        draw_cloud_trail(screen, interpolated_positions, x_range, y_range, width, height)
        
        # Render each cloud with position interpolation and smooth easing
        for i, ellipse_params in enumerate(cloud_ellipses):
            # Get interpolated ellipse parameters
            if i < len(self.weather_system.parcels):
                parcel = self.weather_system.parcels[i]
                if hasattr(parcel, 'prev_x') and hasattr(parcel, 'prev_y'):
                    # Interpolate cloud position with eased alpha for smoother motion
                    cx, cy = ellipse_params[0], ellipse_params[1]
                    interp_x = parcel.prev_x * (1 - eased_alpha) + cx * eased_alpha
                    interp_y = parcel.prev_y * (1 - eased_alpha) + cy * eased_alpha
                    
                    # Create interpolated ellipse parameters
                    interp_ellipse = (interp_x, interp_y) + ellipse_params[2:]
                else:
                    interp_ellipse = ellipse_params
            else:
                interp_ellipse = ellipse_params
            
            # Create and draw cloud surface
            cloud_surface, pos = create_cloud_surface(
                interp_ellipse,
                CFG.DOMAIN_SIZE_M, CFG.AREA_SIZE_KM,
                width, height,
                x_range, y_range
            )
            screen.blit(cloud_surface, pos)
    
    def _draw_cloud_vectors(self, screen, cloud_ellipses, x_range, y_range, width, height):
        """Draw velocity vectors to visualize cloud gliding direction"""
        if not hasattr(self.weather_system, 'parcels') or len(self.weather_system.parcels) == 0:
            return
            
        import pygame
        import math
        
        for i, parcel in enumerate(self.weather_system.parcels):
            if hasattr(parcel, 'x') and hasattr(parcel, 'vx'):
                # Convert position to screen coordinates
                x_km = parcel.x / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                y_km = parcel.y / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                
                x_px, y_px = km_to_screen_coords(x_km, y_km, x_range, y_range, width, height)
                
                # Calculate direction and speed
                speed = math.sqrt(parcel.vx**2 + parcel.vy**2)
                direction = math.degrees(math.atan2(parcel.vy, parcel.vx))
                
                # Draw direction arrow (scaled by speed)
                arrow_length = min(50, speed * 50)
                end_x = x_px + arrow_length * math.cos(math.radians(direction))
                end_y = y_px + arrow_length * math.sin(math.radians(direction))
                
                # Draw arrow
                pygame.draw.line(screen, (255, 0, 0), (x_px, y_px), (end_x, end_y), width=2)
                
                # Draw speed text
                speed_text = f"{direction:.0f}°, {speed*100:.1f}"
                if self.font:
                    speed_surface = self.font.render(speed_text, True, (255, 0, 0))
                    screen.blit(speed_surface, (x_px + 10, y_px - 20))
    
    def render_ui(self, screen, result, width, height, x_range, y_range):
        """Render UI elements efficiently"""
        import pygame
        
        if self.font is None:
            self.font = pygame.font.SysFont('Arial', 16)
            self.large_font = pygame.font.SysFont('Arial', 20, bold=True)
            self.info_panel_bg = pygame.Surface((250, 150), pygame.SRCALPHA)
            self.info_panel_bg.fill((255, 255, 255, 180))
            
        cloud_ellipses = result.get('cloud_ellipses', [])
        panel_coverage = result.get('panel_coverage', {})
        power_output = result.get('power_output', {})
        total_power = power_output.get('total', 0.0)
        total_ac_power = power_output.get('total_ac', 0.0)
        baseline_total = power_output.get('baseline_total', 0.0)
        farm_reduction_pct = power_output.get('farm_reduction_pct', 0.0)
        cloud_cover = result.get('cloud_cover', 0.0)
        timestamp = result.get('time', datetime.now())
        cloud_speed = result.get('cloud_speed', None)
        cloud_direction = result.get('cloud_direction', None)
        confidence = result.get('confidence', 0)
        trajectory_source = result.get('trajectory_source', 'power')
        solar_position = result.get('solar_position', {'elevation': 90, 'azimuth': 180})
        
        time_str = timestamp.strftime("%H:%M") if hasattr(timestamp, 'strftime') else f"{self.current_hour:.1f}h"
        info_dict = {
            "Time": time_str,
            "Sun": f"El={solar_position['elevation']:.1f}°, Az={solar_position['azimuth']:.1f}°",
            "Cloud Cover": f"{cloud_cover:.1f}%",
            "DC Power": f"{total_power:.1f} kW",
            "AC Output": f"{total_ac_power:.1f} kW" if total_ac_power else f"N/A",
            "Farm Reduction": f"{farm_reduction_pct:.1f}%"
        }
        
        if self.use_gl_renderer and OPENGL_AVAILABLE:
            # Initialize GL renderers if not already done
            if self.gl_ui_renderer is None:
                self.gl_ui_renderer = initialize_gl_ui_renderer(screen.get_size())
            
            # Draw solar panels with OpenGL
            if self.gl_panel_renderer is None:
                self.gl_panel_renderer = initialize_gl_panel_renderer(screen.get_size())
            
            # Draw panels using OpenGL if the renderer is available
            if self.gl_panel_renderer:
                affected_panels = gl_draw_solar_panels(
                    screen, self.gl_panel_renderer, self.panel_df, 
                    panel_coverage, power_output, 
                    x_range, y_range, width, height
                )
            else:
                # Fallback to Pygame for panels
                from pygame_rendereres.panel_renderer import draw_solar_panels
                affected_panels = draw_solar_panels(
                    screen, self.panel_df, 
                    panel_coverage, power_output, 
                    x_range, y_range, width, height
                )
            
            # For now, just create panel background with OpenGL and use Pygame for text
            if self.gl_ui_renderer:
                gl_create_info_panel(self.gl_ui_renderer, info_dict, "Simulation Info", (20, 20), 250)
            
            # Use Pygame for text rendering
            self._draw_info_panel(screen, info_dict, "Simulation Info", (20, 20))
            
            if farm_reduction_pct > 30:
                self._draw_warning_bar(screen, farm_reduction_pct, (20, 170), 200, 30)
            
            self._draw_trajectory_info(screen, cloud_speed, cloud_direction, confidence, 
                                     trajectory_source, (20, 210))
            
            affected_panels = [p for p, c in panel_coverage.items() if c > 0]
            self._draw_affected_panels_list(screen, affected_panels, len(self.panel_df), 
                                           power_output, (width - 350, 20))
            
            self._draw_time_slider(screen, self.current_hour, self.power_simulator.sunrise_hour,
                                  self.power_simulator.sunset_hour, (width//2 - 200, height - 40), 400)
        else:
            # Use standard Pygame rendering
            from pygame_rendereres.panel_renderer import draw_solar_panels
            
            affected_panels = draw_solar_panels(
                screen, self.panel_df, 
                panel_coverage, power_output, 
                x_range, y_range, width, height
            )
            
            self._draw_info_panel(screen, info_dict, "Simulation Info", (20, 20))
            
            if farm_reduction_pct > 30:
                self._draw_warning_bar(screen, farm_reduction_pct, (20, 170), 200, 30)
            
            self._draw_trajectory_info(screen, cloud_speed, cloud_direction, confidence, 
                                     trajectory_source, (20, 210))
            
            affected_panels = [p for p, c in panel_coverage.items() if c > 0]
            self._draw_affected_panels_list(screen, affected_panels, len(self.panel_df), 
                                           power_output, (width - 350, 20))
            
            self._draw_time_slider(screen, self.current_hour, self.power_simulator.sunrise_hour,
                                  self.power_simulator.sunset_hour, (width//2 - 200, height - 40), 400)
        
        # Draw help text regardless of renderer
        renderer_text = "OpenGL" if self.use_gl_renderer else "Pygame" 
        help_text = f"Controls: ESC-Quit, ↑↓-Wind Speed, ←→-Movement Speed, G-Toggle Renderer ({renderer_text})"
        help_surface = self.font.render(help_text, True, (0, 0, 0), (255, 255, 255, 180))
        screen.blit(help_surface, (width//2 - help_surface.get_width()//2, height - 70))
        
        # Add cloud vector visualization in debug mode
        if self.debug_mode:
            self._draw_cloud_vectors(screen, cloud_ellipses, x_range, y_range, width, height)
    
    def _draw_info_panel(self, screen, info_dict, title, position):
        """Draw information panel with background"""
        import pygame
        
        padding = 10
        line_height = 24
        num_lines = len(info_dict) + (1 if title else 0)
        panel_height = padding * 2 + line_height * num_lines
        panel_width = 250
        panel_rect = pygame.Rect(position[0], position[1], panel_width, panel_height)
        bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        bg_surface.fill((255, 255, 255, 180))
        screen.blit(bg_surface, position)
        pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1)
        y_offset = position[1] + padding
        if title:
            title_surface = self.large_font.render(title, True, (0, 0, 0))
            screen.blit(title_surface, (position[0] + padding, y_offset))
            y_offset += line_height
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (position[0] + padding, y_offset))
            y_offset += line_height
    
    def _draw_warning_bar(self, screen, reduction_pct, position, width, height):
        """Draw power reduction warning bar"""
        import pygame
        
        bar_rect = pygame.Rect(position[0], position[1], width, height)
        pygame.draw.rect(screen, (220, 220, 220), bar_rect)
        pygame.draw.rect(screen, (0, 0, 0), bar_rect, width=1)
        fill_width = int(min(100, reduction_pct) / 100 * width)
        if fill_width > 0:
            fill_rect = pygame.Rect(position[0], position[1], fill_width, height)
            if reduction_pct < 50:
                color = (255, 120, 0)
            else:
                color = (255, 0, 0)
            pygame.draw.rect(screen, color, fill_rect)
        text = f"Power Reduction Warning: {reduction_pct:.1f}%"
        text_surface = self.font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(position[0] + width//2, position[1] + height//2))
        screen.blit(text_surface, text_rect)
    
    def _draw_trajectory_info(self, screen, cloud_speed, cloud_direction, confidence, source, position):
        """Draw cloud trajectory information with compass"""
        import pygame
        import math
        
        if cloud_speed is None or cloud_direction is None:
            cloud_speed = 10.0
            cloud_direction = 10.0
            confidence = 5.0
            source = 'none'
        
        panel_width = 200
        panel_height = 170
        panel_rect = pygame.Rect(position[0], position[1], panel_width, panel_height)
        bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        bg_surface.fill((255, 255, 255, 180))
        screen.blit(bg_surface, position)
        pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1)
        
        # Title and text info
        title_surface = self.large_font.render("Cloud Movement", True, (0, 0, 0))
        screen.blit(title_surface, (position[0] + 10, position[1] + 10))
        
        speed_text = f"Speed: {cloud_speed:.1f} km/h"
        speed_surface = self.font.render(speed_text, True, (0, 0, 0))
        screen.blit(speed_surface, (position[0] + 10, position[1] + 40))
        
        dir_text = f"Direction: {cloud_direction:.0f}°"
        dir_surface = self.font.render(dir_text, True, (0, 0, 0))
        screen.blit(dir_surface, (position[0] + 10, position[1] + 65))
        
        conf_text = f"Confidence: {int(confidence * 100)}%"
        conf_surface = self.font.render(conf_text, True, (0, 0, 0))
        screen.blit(conf_surface, (position[0] + 10, position[1] + 90))
        
        source_text = f"Source: {source.capitalize()}"
        source_color = (0, 120, 0) if source == 'power' else (0, 0, 120)
        source_surface = self.font.render(source_text, True, source_color)
        screen.blit(source_surface, (position[0] + 10, position[1] + 115))
        
        # Draw compass and arrow
        arrow_center = (position[0] + panel_width // 2, position[1] + panel_height - 30)
        arrow_length = 30
        direction_rad = math.radians(cloud_direction)
        end_x = arrow_center[0] + arrow_length * math.cos(direction_rad)
        end_y = arrow_center[1] - arrow_length * math.sin(direction_rad)
        
        # Compass circle
        pygame.draw.circle(screen, (230, 230, 230), arrow_center, arrow_length + 5)
        pygame.draw.circle(screen, (0, 0, 0), arrow_center, arrow_length + 5, width=1)
        
        # Arrow color based on source
        if source == 'power':
            arrow_color = (0, 150, 0)
        else:
            if confidence > 0.7:
                arrow_color = (0, 0, 150)
            elif confidence > 0.3:
                arrow_color = (150, 150, 0)
            else:
                arrow_color = (150, 0, 0)
        
        # Draw arrow
        pygame.draw.line(screen, arrow_color, arrow_center, (end_x, end_y), width=3)
    
    def _draw_affected_panels_list(self, screen, affected_panels, total_panels, power_output, position):
        """Draw list of affected panels"""
        import pygame
        
        if not affected_panels:
            return
        
        panel_width = 300
        panel_height = 250
        padding = 10
        line_height = 20
        panel_rect = pygame.Rect(position[0], position[1], panel_width, panel_height)
        bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        bg_surface.fill((255, 255, 255, 180))
        screen.blit(bg_surface, position)
        pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1)
        
        affected_count = len(affected_panels)
        affected_pct = affected_count / total_panels * 100 if total_panels > 0 else 0
        title = f"Affected Panels: {affected_count}/{total_panels} ({affected_pct:.1f}%)"
        title_surface = self.large_font.render(title, True, (0, 0, 0))
        screen.blit(title_surface, (position[0] + padding, position[1] + padding))
        
        # Show first 10 panels
        y_pos = position[1] + padding + 30
        for i, panel_id in enumerate(affected_panels[:10]):
            if panel_id not in power_output:
                continue
            power_data = power_output[panel_id]
            baseline = power_data.get('baseline', 0)
            current = power_data.get('final_power', 0)
            if baseline > 0:
                reduction = (baseline - current) / baseline * 100
            else:
                reduction = 0
            text = f"{panel_id:<10}   {reduction:>6.1f}%     {current:.2f} kW"
            if reduction > 50:
                color = (180, 0, 0)
            elif reduction > 20:
                color = (180, 120, 0)
            else:
                color = (0, 120, 0)
            text_surface = self.font.render(text, True, color)
            screen.blit(text_surface, (position[0] + padding, y_pos))
            y_pos += line_height
    
    def _draw_time_slider(self, screen, current_hour, sunrise_hour, sunset_hour, position, width):
        """Draw time slider showing current simulation time"""
        import pygame
        
        height = 20
        day_start = 0
        day_end = 24
        day_length = day_end - day_start
        x_pos = position[0]
        y_pos = position[1]
        
        # Background
        bg_rect = pygame.Rect(x_pos, y_pos, width, height)
        pygame.draw.rect(screen, (220, 220, 220), bg_rect, border_radius=height//2)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect, width=1, border_radius=height//2)
        
        # Current time indicator
        if day_start <= current_hour <= day_end:
            current_x = x_pos + (current_hour - day_start) / day_length * width
            pointer_height = 15
            pointer_width = 10
            pointer_points = [
                (current_x, y_pos - 5),
                (current_x - pointer_width//2, y_pos - 5 - pointer_height),
                (current_x + pointer_width//2, y_pos - 5 - pointer_height)
            ]
            pygame.draw.polygon(screen, (200, 0, 0), pointer_points)
            
            hour = int(current_hour)
            minute = int((current_hour % 1) * 60)
            time_text = f"{hour:02d}:{minute:02d}"
            time_surface = self.font.render(time_text, True, (0, 0, 0))
            time_rect = time_surface.get_rect(center=(current_x, y_pos - 25))
            screen.blit(time_surface, time_rect)