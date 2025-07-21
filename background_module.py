# background_module.py
"""
Complete Environmental Simulation System for Solar Farm Cloud Simulation

Manages environmental conditions with accurate solar position calculation,
atmospheric modeling, and day/night cycles. Designed for thread-safe operation
with the physics worker and provides comprehensive environmental data.

Features:
- Accurate solar position calculation based on time and location
- Atmospheric modeling (temperature, humidity, pressure, air mass)
- Day/night cycle with realistic lighting transitions
- Thread-safe operation for physics worker integration
- Configurable location and environmental parameters
- Integration with weather system and CSV coordinate bounds
"""
import math
import time
import threading
import copy
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np

@dataclass
class EnvironmentState:
    """Complete atmospheric and environmental conditions dataclass."""
    # Time and location
    timestamp: float = field(default_factory=time.time)
    simulation_time: float = 0.0  # Simulation seconds
    current_hour: float = 12.0    # Local time (0-24)
    day_of_year: int = field(default_factory=lambda: datetime.now().timetuple().tm_yday)
    latitude: float = 6.9271      # Degrees north
    longitude: float = 79.8612    # Degrees east
    timezone_offset: float = 5.5  # Hours from UTC
    
    # Solar position
    solar_elevation: float = 45.0      # Degrees above horizon
    solar_azimuth: float = 180.0       # Degrees from north (clockwise)
    solar_declination: float = 0.0     # Solar declination angle
    hour_angle: float = 0.0            # Hour angle
    air_mass: float = 1.0              # Atmospheric air mass factor
    
    # Atmospheric conditions
    ambient_temperature: float = 30.0   # °C
    humidity: float = 65.0              # Relative humidity (%)
    pressure: float = 1013.25           # Atmospheric pressure (hPa)
    atmospheric_clarity: float = 0.75   # Clear sky index (0-1)
    turbidity: float = 2.5              # Atmospheric turbidity
    water_vapor: float = 1.5            # Water vapor (cm)
    ozone: float = 0.3                  # Ozone (atm-cm)
    
    # Lighting and visual conditions
    ambient_light: float = 0.8          # Ambient light level (0-1)
    shadow_sharpness: float = 0.7       # Shadow sharpness factor (0-1)
    sky_brightness: float = 0.9         # Sky brightness (0-1)
    
    # Environmental dynamics
    wind_speed: float = 5.0             # Wind speed (m/s)
    wind_direction: float = 90.0        # Wind direction (degrees)
    cloud_coverage: float = 25.0        # Cloud coverage percentage
    
    # Radiation components
    direct_normal_irradiance: float = 800.0   # W/m²
    diffuse_horizontal_irradiance: float = 200.0  # W/m²
    global_horizontal_irradiance: float = 1000.0  # W/m²
    
    # Day/night cycle
    is_daytime: bool = True
    sunrise_hour: float = 6.0
    sunset_hour: float = 18.0
    daylight_factor: float = 1.0        # 0-1 based on sun position
    
    # Simulation metadata
    time_scale: float = 1.0             # Time acceleration factor
    location_name: str = "Negombo, Sri Lanka"

class BackgroundModule:
    """
    Complete environmental simulation system managing atmospheric conditions,
    solar position, and day/night cycles with thread-safe operation.
    """
    
    def __init__(self, latitude: float = 6.9271, longitude: float = 79.8612,
                 timezone_offset: float = 5.5, elevation_m: float = 10.0,
                 time_scale: float = 1.0):
        """
        Initialize the background environmental simulation module.
        
        Args:
            latitude: Site latitude in degrees (negative for south)
            longitude: Site longitude in degrees (negative for west)  
            timezone_offset: Hours from UTC (positive for east)
            elevation_m: Site elevation in meters above sea level
            time_scale: Time acceleration factor (1.0 = real time)
        """
        # Location and time configuration
        self.latitude = latitude
        self.longitude = longitude
        self.timezone_offset = timezone_offset
        self.elevation_m = elevation_m
        self.time_scale = time_scale
        
        # Environmental state
        self.environment_state = EnvironmentState(
            latitude=latitude,
            longitude=longitude,
            timezone_offset=timezone_offset,
            time_scale=time_scale
        )
        
        # Threading support
        self.thread_safe_mode = False
        self._lock = threading.RLock()
        
        # Time management
        self.simulation_start_time = time.time()
        self.last_update_time = time.time()
        self.simulation_time = 0.0
        
        # Environmental parameter ranges for realistic variations
        self.temperature_range = (25.0, 35.0)  # °C for tropical location
        self.humidity_range = (60.0, 90.0)     # % for tropical climate
        self.pressure_range = (1008.0, 1018.0) # hPa typical range
        
        # Solar position calculation cache
        self.solar_position_cache = {}
        self.cache_precision = 0.1  # Cache solar positions to 0.1 hour precision
        
        # Atmospheric model parameters
        self.atmospheric_params = {
            'rayleigh_scattering': 0.0903,
            'aerosol_optical_depth': 0.1,
            'water_vapor_absorption': 0.2385,
            'ozone_absorption': 1.0,
            'mixed_gas_absorption': 0.0127
        }
        
        # Day/night cycle parameters
        self.twilight_duration = 1.0  # Hours for sunrise/sunset transition
        
        # Performance tracking
        self.update_count = 0
        self.last_performance_check = time.time()
        
        # Initialize with current conditions
        self._update_solar_position()
        self._update_atmospheric_conditions()
        self._update_day_night_cycle()
        
        print(f"BackgroundModule initialized:")
        print(f"  Location: {latitude:.3f}°N, {longitude:.3f}°E")
        print(f"  Elevation: {elevation_m}m")
        print(f"  Timezone: UTC{timezone_offset:+.1f}")
        print(f"  Time scale: {time_scale}x")
        print(f"  Current conditions: {self.environment_state.ambient_temperature:.1f}°C, "
              f"{self.environment_state.humidity:.0f}% humidity")
    
    def set_thread_safe_mode(self, enabled: bool):
        """Enable or disable thread-safe mode for worker thread operation."""
        self.thread_safe_mode = enabled
        if enabled:
            print("BackgroundModule: Thread-safe mode enabled")
    
    def update(self, dt: float):
        """
        Update environmental conditions for the given time step.
        
        Args:
            dt: Delta time in seconds
        """
        if self.thread_safe_mode:
            with self._lock:
                self._update_internal(dt)
        else:
            self._update_internal(dt)
    
    def _update_internal(self, dt: float):
        """Internal update implementation with environmental progression."""
        current_time = time.time()
        
        # Update simulation time with time scale
        self.simulation_time += dt * self.time_scale
        
        # Update current hour (0-24) based on simulation time
        hours_elapsed = self.simulation_time / 3600.0
        self.environment_state.current_hour = (12.0 + hours_elapsed) % 24.0  # Start at noon
        self.environment_state.simulation_time = self.simulation_time
        self.environment_state.timestamp = current_time
        
        # Update day of year if significant time has passed
        if self.time_scale > 24.0:  # If accelerated enough to matter
            days_elapsed = hours_elapsed / 24.0
            start_day = datetime.now().timetuple().tm_yday
            self.environment_state.day_of_year = int(start_day + days_elapsed) % 365 + 1
        
        # Update environmental components
        self._update_solar_position()
        self._update_atmospheric_conditions()
        self._update_day_night_cycle()
        self._update_environmental_variations(dt)
        
        # Update performance tracking
        self.update_count += 1
        self.last_update_time = current_time
    
    def _update_solar_position(self):
        """Calculate accurate solar position based on time and location."""
        # Create cache key for solar position (precision to 0.1 hour)
        cache_key = (
            round(self.environment_state.current_hour / self.cache_precision) * self.cache_precision,
            self.environment_state.day_of_year,
            round(self.latitude, 2),
            round(self.longitude, 2)
        )
        
        # Check cache first
        if cache_key in self.solar_position_cache:
            cached_result = self.solar_position_cache[cache_key]
            self.environment_state.solar_elevation = cached_result['elevation']
            self.environment_state.solar_azimuth = cached_result['azimuth']
            self.environment_state.solar_declination = cached_result['declination']
            self.environment_state.hour_angle = cached_result['hour_angle']
            self.environment_state.air_mass = cached_result['air_mass']
            return
        
        # Calculate solar position from scratch
        hour = self.environment_state.current_hour
        day_of_year = self.environment_state.day_of_year
        lat = self.latitude
        lon = self.longitude
        
        # Solar declination angle (varies seasonally)
        declination_rad = math.radians(23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365)))
        declination = math.degrees(declination_rad)
        
        # Hour angle (15 degrees per hour from solar noon)
        solar_time = hour + lon / 15.0 - self.timezone_offset
        hour_angle_rad = math.radians(15 * (solar_time - 12))
        hour_angle = math.degrees(hour_angle_rad)
        
        # Solar elevation angle
        lat_rad = math.radians(lat)
        sin_elevation = (math.sin(declination_rad) * math.sin(lat_rad) + 
                        math.cos(declination_rad) * math.cos(lat_rad) * math.cos(hour_angle_rad))
        elevation_rad = math.asin(max(-1, min(1, sin_elevation)))
        elevation = math.degrees(elevation_rad)
        
        # Solar azimuth angle
        if elevation > 0:
            cos_azimuth = ((math.sin(declination_rad) * math.cos(lat_rad) - 
                           math.cos(declination_rad) * math.sin(lat_rad) * math.cos(hour_angle_rad)) / 
                           math.cos(elevation_rad))
            cos_azimuth = max(-1, min(1, cos_azimuth))
            azimuth_rad = math.acos(cos_azimuth)
            
            if hour_angle > 0:  # Afternoon
                azimuth = math.degrees(azimuth_rad)
            else:  # Morning
                azimuth = 360 - math.degrees(azimuth_rad)
        else:
            azimuth = 0  # Sun below horizon
        
        # Atmospheric air mass calculation
        if elevation > 0:
            air_mass = 1 / (math.sin(elevation_rad) + 0.50572 * ((elevation + 6.07995) ** -1.6364))
            # Apply elevation correction
            air_mass *= math.exp(-self.elevation_m / 8400.0)  # Scale height ~8.4km
        else:
            air_mass = float('inf')
        
        # Limit air mass to reasonable values
        air_mass = min(air_mass, 38.0)
        
        # Store results
        result = {
            'elevation': max(0, elevation),
            'azimuth': azimuth,
            'declination': declination,
            'hour_angle': hour_angle,
            'air_mass': air_mass
        }
        
        # Cache result
        self.solar_position_cache[cache_key] = result
        
        # Limit cache size
        if len(self.solar_position_cache) > 1000:
            # Remove oldest entries
            keys_to_remove = list(self.solar_position_cache.keys())[:500]
            for key in keys_to_remove:
                del self.solar_position_cache[key]
        
        # Update environment state
        self.environment_state.solar_elevation = result['elevation']
        self.environment_state.solar_azimuth = result['azimuth']
        self.environment_state.solar_declination = result['declination']
        self.environment_state.hour_angle = result['hour_angle']
        self.environment_state.air_mass = result['air_mass']
    
    def _update_atmospheric_conditions(self):
        """Update atmospheric conditions based on solar position and time."""
        elevation = self.environment_state.solar_elevation
        air_mass = self.environment_state.air_mass
        
        # Calculate solar irradiance components
        if elevation > 0:
            # Solar constant with Earth-Sun distance correction
            day_of_year = self.environment_state.day_of_year
            B = 2 * math.pi * (day_of_year - 1) / 365
            distance_correction = (1.000110 + 0.034221 * math.cos(B) + 
                                 0.001280 * math.sin(B) + 0.000719 * math.cos(2*B) + 
                                 0.000077 * math.sin(2*B))
            
            extraterrestrial_irradiance = 1367.0 * distance_correction
            
            # Atmospheric transmission calculations
            elevation_rad = math.radians(elevation)
            
            # Rayleigh scattering
            tau_r = math.exp(-self.atmospheric_params['rayleigh_scattering'] * 
                           (air_mass ** 0.84) * (1 + air_mass - (air_mass ** 1.01)))
            
            # Aerosol extinction
            tau_a = math.exp(-air_mass * self.atmospheric_params['aerosol_optical_depth'])
            
            # Water vapor absorption
            tau_w = math.exp(-self.atmospheric_params['water_vapor_absorption'] * 
                           self.environment_state.water_vapor * air_mass / 
                           (1 + 20.07 * self.environment_state.water_vapor * air_mass) ** 0.45)
            
            # Ozone absorption
            tau_o = math.exp(-air_mass * self.environment_state.ozone * 
                           (1 + 139.48 * self.environment_state.ozone) ** -0.3035 - 
                           0.02715 * self.environment_state.ozone / 
                           (1 + 0.0316 * self.environment_state.ozone))
            
            # Mixed gas absorption
            tau_g = math.exp(-self.atmospheric_params['mixed_gas_absorption'] * (air_mass ** 0.26))
            
            # Direct normal irradiance
            dni = extraterrestrial_irradiance * tau_r * tau_a * tau_w * tau_o * tau_g
            
            # Diffuse horizontal irradiance (simplified model)
            dhi = extraterrestrial_irradiance * math.sin(elevation_rad) * 0.095 * \
                  (1 - tau_r * tau_a * tau_w * tau_o * tau_g)
            
            # Global horizontal irradiance
            ghi = dni * math.sin(elevation_rad) + dhi
            
            # Update irradiance values
            self.environment_state.direct_normal_irradiance = max(0, dni)
            self.environment_state.diffuse_horizontal_irradiance = max(0, dhi)
            self.environment_state.global_horizontal_irradiance = max(0, ghi)
            
            # Calculate atmospheric clarity
            clear_sky_ghi = extraterrestrial_irradiance * math.sin(elevation_rad) * 0.75
            self.environment_state.atmospheric_clarity = min(1.0, ghi / clear_sky_ghi if clear_sky_ghi > 0 else 0.75)
            
        else:
            # Nighttime - no solar irradiance
            self.environment_state.direct_normal_irradiance = 0.0
            self.environment_state.diffuse_horizontal_irradiance = 0.0
            self.environment_state.global_horizontal_irradiance = 0.0
            self.environment_state.atmospheric_clarity = 0.0
        
        # Update shadow sharpness based on atmospheric conditions
        if elevation > 5:  # Only during significant daylight
            base_sharpness = 0.9
            clarity_factor = self.environment_state.atmospheric_clarity
            humidity_factor = 1.0 - (self.environment_state.humidity - 50.0) / 100.0  # Higher humidity = softer shadows
            self.environment_state.shadow_sharpness = base_sharpness * clarity_factor * max(0.3, humidity_factor)
        else:
            self.environment_state.shadow_sharpness = 0.0
    
    def _update_day_night_cycle(self):
        """Update day/night cycle and lighting conditions."""
        elevation = self.environment_state.solar_elevation
        hour = self.environment_state.current_hour
        
        # Calculate sunrise and sunset times (simplified for tropical location)
        day_of_year = self.environment_state.day_of_year
        
        # Seasonal variation (less pronounced near equator)
        seasonal_variation = 0.5 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        base_day_length = 12.0  # Hours for equatorial location
        
        sunrise_hour = 6.0 - seasonal_variation
        sunset_hour = 18.0 + seasonal_variation
        
        self.environment_state.sunrise_hour = sunrise_hour
        self.environment_state.sunset_hour = sunset_hour
        
        # Determine if it's daytime
        self.environment_state.is_daytime = sunrise_hour <= hour <= sunset_hour
        
        # Calculate daylight factor (0-1) with smooth transitions
        if elevation > 0:
            # Full daylight
            daylight_factor = min(1.0, elevation / 45.0)
        elif hour < sunrise_hour - self.twilight_duration or hour > sunset_hour + self.twilight_duration:
            # Full nighttime
            daylight_factor = 0.0
        else:
            # Twilight transition
            if hour < sunrise_hour:
                # Dawn twilight
                twilight_progress = (sunrise_hour - hour) / self.twilight_duration
                daylight_factor = max(0.0, 1.0 - twilight_progress) * 0.3
            else:
                # Dusk twilight
                twilight_progress = (hour - sunset_hour) / self.twilight_duration
                daylight_factor = max(0.0, 1.0 - twilight_progress) * 0.3
        
        self.environment_state.daylight_factor = daylight_factor
        
        # Update ambient light based on daylight factor and atmospheric conditions
        base_ambient = 0.1 + 0.9 * daylight_factor
        clarity_bonus = self.environment_state.atmospheric_clarity * 0.1
        self.environment_state.ambient_light = min(1.0, base_ambient + clarity_bonus)
        
        # Update sky brightness
        if daylight_factor > 0.1:
            sky_brightness = 0.3 + 0.7 * daylight_factor
        else:
            sky_brightness = 0.05 + 0.25 * daylight_factor  # Moonlight and stars
        
        self.environment_state.sky_brightness = sky_brightness
    
    def _update_environmental_variations(self, dt: float):
        """Update environmental parameters with realistic variations."""
        # Temperature variation based on solar irradiance and time of day
        base_temp = 25.0 + 10.0 * self.environment_state.daylight_factor
        
        # Add daily temperature cycle
        hour_factor = math.sin(2 * math.pi * (self.environment_state.current_hour - 6) / 24)
        temp_variation = 3.0 * hour_factor
        
        # Add gradual drift
        time_factor = math.sin(self.simulation_time * 0.001) * 2.0
        
        target_temp = base_temp + temp_variation + time_factor
        target_temp = max(self.temperature_range[0], min(self.temperature_range[1], target_temp))
        
        # Smooth temperature changes
        temp_change_rate = 1.0  # degrees per hour
        max_change = temp_change_rate * dt / 3600.0
        temp_diff = target_temp - self.environment_state.ambient_temperature
        temp_change = max(-max_change, min(max_change, temp_diff))
        
        self.environment_state.ambient_temperature += temp_change
        
        # Humidity variation (inverse relationship with temperature during day)
        base_humidity = 75.0
        temp_humidity_factor = -0.5 * (self.environment_state.ambient_temperature - 27.0)
        time_humidity_factor = 10.0 * math.sin(self.simulation_time * 0.002 + math.pi)
        
        target_humidity = base_humidity + temp_humidity_factor + time_humidity_factor
        target_humidity = max(self.humidity_range[0], min(self.humidity_range[1], target_humidity))
        
        # Smooth humidity changes
        humidity_change_rate = 5.0  # percent per hour
        max_humidity_change = humidity_change_rate * dt / 3600.0
        humidity_diff = target_humidity - self.environment_state.humidity
        humidity_change = max(-max_humidity_change, min(max_humidity_change, humidity_diff))
        
        self.environment_state.humidity += humidity_change
        
        # Pressure variation (realistic daily cycle)
        base_pressure = 1013.25
        pressure_cycle = 2.0 * math.sin(2 * math.pi * self.environment_state.current_hour / 24)
        pressure_drift = 1.0 * math.sin(self.simulation_time * 0.0005)
        
        target_pressure = base_pressure + pressure_cycle + pressure_drift
        self.environment_state.pressure = max(self.pressure_range[0], 
                                            min(self.pressure_range[1], target_pressure))
        
        # Update turbidity based on humidity and time
        base_turbidity = 2.5
        humidity_turbidity = (self.environment_state.humidity - 60.0) / 100.0
        time_turbidity = 0.5 * math.sin(self.simulation_time * 0.001)
        
        self.environment_state.turbidity = max(1.5, min(4.0, 
            base_turbidity + humidity_turbidity + time_turbidity))
    
    def get_state(self) -> EnvironmentState:
        """
        Get complete environmental data.
        
        Returns:
            Deep copy of current EnvironmentState
        """
        if self.thread_safe_mode:
            with self._lock:
                return copy.deepcopy(self.environment_state)
        else:
            return copy.deepcopy(self.environment_state)
    
    def set_location(self, latitude: float, longitude: float, timezone_offset: float = None):
        """
        Set simulation location and update environmental conditions.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees  
            timezone_offset: Hours from UTC (auto-calculated if None)
        """
        if self.thread_safe_mode:
            with self._lock:
                self._set_location_internal(latitude, longitude, timezone_offset)
        else:
            self._set_location_internal(latitude, longitude, timezone_offset)
    
    def _set_location_internal(self, latitude: float, longitude: float, timezone_offset: Optional[float]):
        """Internal location setting implementation."""
        self.latitude = latitude
        self.longitude = longitude
        
        if timezone_offset is not None:
            self.timezone_offset = timezone_offset
        else:
            # Auto-calculate timezone from longitude
            self.timezone_offset = round(longitude / 15.0)
        
        # Update environment state
        self.environment_state.latitude = latitude
        self.environment_state.longitude = longitude
        self.environment_state.timezone_offset = self.timezone_offset
        
        # Clear solar position cache
        self.solar_position_cache.clear()
        
        # Recalculate current conditions
        self._update_solar_position()
        self._update_atmospheric_conditions()
        self._update_day_night_cycle()
        
        print(f"Location updated: {latitude:.3f}°N, {longitude:.3f}°E, UTC{self.timezone_offset:+.1f}")
    
    def set_time_scale(self, time_scale: float):
        """
        Set time acceleration factor.
        
        Args:
            time_scale: Time scale factor (1.0 = real time, 60.0 = 1 minute per second)
        """
        if self.thread_safe_mode:
            with self._lock:
                self.time_scale = max(0.1, min(3600.0, time_scale))
                self.environment_state.time_scale = self.time_scale
        else:
            self.time_scale = max(0.1, min(3600.0, time_scale))
            self.environment_state.time_scale = self.time_scale
        
        print(f"Time scale set to {self.time_scale}x")
    
    def set_atmospheric_parameters(self, **kwargs):
        """
        Set atmospheric model parameters.
        
        Args:
            **kwargs: Atmospheric parameters (turbidity, water_vapor, ozone, etc.)
        """
        if self.thread_safe_mode:
            with self._lock:
                self._set_atmospheric_parameters_internal(kwargs)
        else:
            self._set_atmospheric_parameters_internal(kwargs)
    
    def _set_atmospheric_parameters_internal(self, params: Dict[str, Any]):
        """Internal atmospheric parameters setting."""
        for key, value in params.items():
            if hasattr(self.environment_state, key):
                setattr(self.environment_state, key, value)
            elif key in self.atmospheric_params:
                self.atmospheric_params[key] = value
        
        # Recalculate atmospheric conditions
        self._update_atmospheric_conditions()
    
    def get_solar_irradiance_components(self) -> Tuple[float, float, float]:
        """
        Get current solar irradiance components.
        
        Returns:
            Tuple of (direct_normal, diffuse_horizontal, global_horizontal) in W/m²
        """
        state = self.get_state()
        return (state.direct_normal_irradiance, 
                state.diffuse_horizontal_irradiance, 
                state.global_horizontal_irradiance)
    
    def is_daylight(self, threshold: float = 5.0) -> bool:
        """
        Check if there is sufficient daylight.
        
        Args:
            threshold: Minimum solar elevation angle for daylight
            
        Returns:
            True if solar elevation above threshold
        """
        return self.environment_state.solar_elevation > threshold
    
    def get_air_mass(self) -> float:
        """Get current atmospheric air mass factor."""
        return self.environment_state.air_mass
    
    def get_daylight_factor(self) -> float:
        """Get current daylight factor (0-1)."""
        return self.environment_state.daylight_factor
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get environmental simulation performance statistics."""
        current_time = time.time()
        uptime = current_time - self.simulation_start_time
        
        return {
            'update_count': self.update_count,
            'uptime_seconds': uptime,
            'updates_per_second': self.update_count / uptime if uptime > 0 else 0,
            'cache_size': len(self.solar_position_cache),
            'simulation_time': self.simulation_time,
            'time_scale': self.time_scale,
            'thread_safe_mode': self.thread_safe_mode,
            'current_conditions': {
                'solar_elevation': self.environment_state.solar_elevation,
                'temperature': self.environment_state.ambient_temperature,
                'humidity': self.environment_state.humidity,
                'is_daytime': self.environment_state.is_daytime
            }
        }
    
    def reset_simulation_time(self):
        """Reset simulation time to start."""
        if self.thread_safe_mode:
            with self._lock:
                self.simulation_time = 0.0
                self.environment_state.simulation_time = 0.0
                self.environment_state.current_hour = 12.0  # Reset to noon
        else:
            self.simulation_time = 0.0
            self.environment_state.simulation_time = 0.0
            self.environment_state.current_hour = 12.0
    
    def cleanup(self):
        """Clean up resources and caches."""
        if self.thread_safe_mode:
            with self._lock:
                self.solar_position_cache.clear()
        else:
            self.solar_position_cache.clear()
        
        print("BackgroundModule cleanup completed")

# Factory functions for easy creation
def create_background_module(latitude: float = 6.9271, longitude: float = 79.8612,
                           **kwargs) -> BackgroundModule:
    """
    Factory function to create a BackgroundModule.
    
    Args:
        latitude: Site latitude
        longitude: Site longitude
        **kwargs: Additional parameters
        
    Returns:
        Configured BackgroundModule instance
    """
    return BackgroundModule(latitude, longitude, **kwargs)

def create_tropical_background_module(**kwargs) -> BackgroundModule:
    """Create a BackgroundModule optimized for tropical locations."""
    module = BackgroundModule(latitude=6.9271, longitude=79.8612, **kwargs)
    
    # Set tropical-specific parameters
    module.set_atmospheric_parameters(
        turbidity=3.0,          # Higher atmospheric turbidity
        water_vapor=3.0,        # Higher humidity
        humidity=75.0           # High base humidity
    )
    
    return module

# Testing and validation functions
def test_background_module():
    """Test the BackgroundModule functionality."""
    print("Testing BackgroundModule...")
    
    # Create module
    module = create_background_module()
    
    # Test basic functionality
    print(f"✓ Module created at {module.latitude:.3f}°N, {module.longitude:.3f}°E")
    
    # Test state retrieval
    state = module.get_state()
    print(f"✓ Current state: {state.ambient_temperature:.1f}°C, "
          f"Solar elevation: {state.solar_elevation:.1f}°")
    
    # Test time progression
    for i in range(5):
        module.update(3600.0)  # 1 hour steps
        state = module.get_state()
        print(f"  Hour {state.current_hour:.1f}: "
              f"Temp {state.ambient_temperature:.1f}°C, "
              f"Solar {state.solar_elevation:.1f}°, "
              f"Daylight: {'Yes' if state.is_daytime else 'No'}")
    
    # Test location change
    module.set_location(40.7128, -74.0060, -5.0)  # New York
    state = module.get_state()
    print(f"✓ Location changed: Solar elevation now {state.solar_elevation:.1f}°")
    
    # Test time scale
    module.set_time_scale(60.0)  # 1 minute real time = 1 hour sim time
    module.update(60.0)  # 1 minute = 1 hour simulation
    state = module.get_state()
    print(f"✓ Time scale test: Hour advanced to {state.current_hour:.1f}")
    
    # Test performance
    stats = module.get_performance_stats()
    print(f"✓ Performance: {stats['updates_per_second']:.1f} updates/sec, "
          f"Cache size: {stats['cache_size']}")
    
    # Test thread safety
    module.set_thread_safe_mode(True)
    for i in range(3):
        module.update(1.0)
    print("✓ Thread-safe mode tested")
    
    # Cleanup
    module.cleanup()
    print("✓ BackgroundModule test completed successfully")

def benchmark_background_module():
    """Benchmark the BackgroundModule performance."""
    print("Benchmarking BackgroundModule...")
    
    module = create_background_module()
    
    # Benchmark update performance
    iterations = 1000
    start_time = time.time()
    
    for i in range(iterations):
        module.update(1.0)  # 1 second steps
    
    end_time = time.time()
    total_time = end_time - start_time
    updates_per_second = iterations / total_time
    
    print(f"Update performance: {updates_per_second:.1f} updates/second")
    print(f"Average update time: {total_time/iterations*1000:.2f} ms")
    
    # Benchmark solar position calculation
    start_time = time.time()
    for hour in np.arange(0, 24, 0.1):  # Every 6 minutes
        module.environment_state.current_hour = hour
        module._update_solar_position()
    
    end_time = time.time()
    calc_time = end_time - start_time
    
    print(f"Solar position calculations: {calc_time*1000:.2f} ms for 24 hours")
    
    # Test cache performance
    cache_hits = 0
    cache_misses = 0
    
    for i in range(100):
        initial_cache_size = len(module.solar_position_cache)
        module._update_solar_position()
        final_cache_size = len(module.solar_position_cache)
        
        if final_cache_size > initial_cache_size:
            cache_misses += 1
        else:
            cache_hits += 1
    
    print(f"Cache performance: {cache_hits}% hits, {cache_misses}% misses")
    
    # Get final stats
    stats = module.get_performance_stats()
    print(f"Final stats: {stats}")
    
    module.cleanup()
    print("Benchmark completed")

if __name__ == "__main__":
    test_background_module()
    benchmark_background_module()