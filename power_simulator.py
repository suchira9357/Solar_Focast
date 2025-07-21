# power_simulator.py
"""
Thread-Optimized Power Simulator for Solar Farm Simulation
Calculates power output with vectorized operations for threading support
"""
import math
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from functools import lru_cache
import warnings

# Suppress warnings for cleaner threading output
warnings.filterwarnings("ignore", category=RuntimeWarning)

@dataclass
class SolarPosition:
    """Solar position data with caching for performance."""
    elevation: float  # degrees above horizon
    azimuth: float    # degrees from north (clockwise)
    declination: float  # solar declination angle
    hour_angle: float   # hour angle
    air_mass: float     # atmospheric air mass factor

@dataclass  
class AtmosphericConditions:
    """Atmospheric conditions affecting solar irradiance."""
    clearness_index: float = 0.75  # Clear sky index (0-1)
    turbidity: float = 2.5          # Atmospheric turbidity
    pressure: float = 1013.25       # Atmospheric pressure (hPa)
    water_vapor: float = 1.5        # Water vapor (cm)
    ozone: float = 0.3              # Ozone (atm-cm)
    aerosol_optical_depth: float = 0.1  # Aerosol optical depth

# Vectorized helper functions for GIL release
@lru_cache(maxsize=24*4)  # Cache for 15-minute intervals over 24 hours
def _cached_solar_position(hour: float, day_of_year: int, latitude: float, 
                          longitude: float, timezone_offset: float) -> Dict[str, float]:
    """Cached solar position calculation with GIL-releasing math operations."""
    # Solar declination angle (varies seasonally)
    declination = math.radians(23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365)))
    
    # Hour angle (15 degrees per hour from solar noon)
    solar_time = hour + longitude / 15 - timezone_offset
    hour_angle = math.radians(15 * (solar_time - 12))
    
    # Solar elevation angle
    latitude_rad = math.radians(latitude)
    sin_elevation = (math.sin(declination) * math.sin(latitude_rad) + 
                    math.cos(declination) * math.cos(latitude_rad) * math.cos(hour_angle))
    elevation_rad = math.asin(max(-1, min(1, sin_elevation)))
    elevation = math.degrees(elevation_rad)
    
    # Solar azimuth angle
    if elevation > 0:
        cos_azimuth = ((math.sin(declination) * math.cos(latitude_rad) - 
                       math.cos(declination) * math.sin(latitude_rad) * math.cos(hour_angle)) / 
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
    else:
        air_mass = float('inf')
    
    return {
        "elevation": max(0, elevation),
        "azimuth": azimuth,
        "declination": math.degrees(declination),
        "hour_angle": math.degrees(hour_angle),
        "air_mass": min(air_mass, 38)  # Cap air mass at 38
    }

def _vectorized_irradiance_calculation(solar_elevations: np.ndarray, 
                                     air_masses: np.ndarray,
                                     day_of_year: int,
                                     atmospheric_conditions: AtmosphericConditions) -> np.ndarray:
    """
    Vectorized irradiance calculation for multiple solar positions.
    Releases GIL through pure NumPy operations.
    
    Args:
        solar_elevations: Array of solar elevation angles in degrees
        air_masses: Array of air mass values
        day_of_year: Day of year (1-365)
        atmospheric_conditions: Atmospheric conditions
        
    Returns:
        Array of global horizontal irradiance values
    """
    # Solar constant and distance correction
    solar_constant = 1367.0  # W/m²
    B = 2 * np.pi * (day_of_year - 1) / 365
    distance_correction = (1.000110 + 0.034221 * np.cos(B) + 0.001280 * np.sin(B) + 
                          0.000719 * np.cos(2*B) + 0.000077 * np.sin(2*B))
    
    et_irradiance = solar_constant * distance_correction
    
    # Mask for above-horizon sun positions
    valid_mask = solar_elevations > 0
    
    # Initialize output arrays
    ghi = np.zeros_like(solar_elevations, dtype=np.float32)
    
    if not np.any(valid_mask):
        return ghi
    
    # Extract valid values for vectorized computation
    valid_elevations = solar_elevations[valid_mask]
    valid_air_masses = air_masses[valid_mask]
    
    # Convert to radians for trigonometric functions
    elevation_rad = np.radians(valid_elevations)
    
    # Atmospheric transmission calculations (vectorized)
    # Rayleigh scattering
    tr = np.exp(-0.0903 * (valid_air_masses ** 0.84) * 
                (1 + valid_air_masses - (valid_air_masses ** 1.01)))
    
    # Aerosol extinction
    ta = np.exp(-valid_air_masses * atmospheric_conditions.aerosol_optical_depth)
    
    # Water vapor absorption
    tw = np.exp(-0.2385 * atmospheric_conditions.water_vapor * valid_air_masses / 
                np.power(1 + 20.07 * atmospheric_conditions.water_vapor * valid_air_masses, 0.45))
    
    # Ozone absorption
    to = np.exp(-valid_air_masses * atmospheric_conditions.ozone * 
                np.power(1 + 139.48 * atmospheric_conditions.ozone, -0.3035) - 
                0.02715 * atmospheric_conditions.ozone / 
                (1 + 0.0316 * atmospheric_conditions.ozone))
    
    # Mixed gas absorption
    tg = np.exp(-0.0127 * np.power(valid_air_masses, 0.26))
    
    # Direct normal irradiance
    dni = et_irradiance * tr * ta * tw * to * tg
    
    # Diffuse horizontal irradiance (simplified model)
    dhi = et_irradiance * np.sin(elevation_rad) * 0.095 * (1 - tr * ta * tw * to * tg)
    
    # Global horizontal irradiance
    valid_ghi = dni * np.sin(elevation_rad) + dhi
    
    # Assign computed values back to full array
    ghi[valid_mask] = np.maximum(0, valid_ghi)
    
    return ghi

def _vectorized_power_calculation(poa_irradiances: np.ndarray, 
                                panel_capacities: np.ndarray,
                                coverages: np.ndarray,
                                system_efficiency: float = 0.85,
                                temperature_coefficient: float = -0.004,
                                reference_temp: float = 25.0,
                                ambient_temp: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized power calculation for multiple panels.
    Optimized for threading with GIL-releasing NumPy operations.
    
    Args:
        poa_irradiances: Array of plane-of-array irradiances (W/m²)
        panel_capacities: Array of panel capacities (kW)
        coverages: Array of cloud coverage values (0-1)
        system_efficiency: Overall system efficiency
        temperature_coefficient: Power temperature coefficient (%/°C)
        reference_temp: Reference temperature (°C)
        ambient_temp: Ambient temperature (°C)
        
    Returns:
        Tuple of (dc_powers, ac_powers) arrays
    """
    # Apply cloud coverage to irradiance
    actual_irradiances = poa_irradiances * (1.0 - coverages)
    
    # Calculate cell temperatures (vectorized NOCT model)
    noct = 45.0  # °C
    noct_irradiance = 800.0  # W/m²
    cell_temps = ambient_temp + (noct - 20) * (actual_irradiances / noct_irradiance)
    
    # Standard test conditions
    stc_irradiance = 1000.0  # W/m²
    
    # Irradiance effect (linear)
    irradiance_factors = actual_irradiances / stc_irradiance
    
    # Temperature effect
    temp_diffs = cell_temps - reference_temp
    temp_factors = 1.0 + temperature_coefficient * temp_diffs
    
    # DC power calculation (vectorized)
    dc_powers = panel_capacities * irradiance_factors * temp_factors * system_efficiency
    dc_powers = np.maximum(0, dc_powers)
    
    # AC power calculation with efficiency curve (vectorized)
    # Simplified inverter efficiency based on power levels
    efficiency_mask_low = dc_powers < 0.1
    efficiency_mask_med = (dc_powers >= 0.1) & (dc_powers < 1.0)
    efficiency_mask_high = dc_powers >= 1.0
    
    inverter_efficiencies = np.ones_like(dc_powers) * 0.95  # Default high efficiency
    inverter_efficiencies[efficiency_mask_low] = 0.85       # Low power efficiency
    inverter_efficiencies[efficiency_mask_med] = 0.90       # Medium power efficiency
    
    ac_powers = dc_powers * inverter_efficiencies
    
    return dc_powers, ac_powers

class ThreadSafePowerSimulator:
    """
    Thread-optimized power simulator with vectorized operations and minimal locking.
    Designed for efficient operation in the PhysicsWorker thread.
    """
    
    def __init__(self, panel_df=None, latitude: float = 6.9271, longitude: float = 79.8612, 
                 timezone_offset: float = 5.5, panel_tilt: float = 0.0, panel_azimuth: float = 180.0):
        """
        Initialize thread-safe power simulator.
        
        Args:
            panel_df: DataFrame with panel information
            latitude: Site latitude in degrees (negative for south)
            longitude: Site longitude in degrees (negative for west)
            timezone_offset: Hours from UTC (positive for east)
            panel_tilt: Panel tilt angle in degrees (0=horizontal, 90=vertical)
            panel_azimuth: Panel azimuth in degrees (0=north, 180=south)
        """
        self.panel_df = panel_df
        self.latitude = latitude
        self.longitude = longitude
        self.timezone_offset = timezone_offset
        self.panel_tilt = math.radians(panel_tilt)
        self.panel_azimuth = math.radians(panel_azimuth)
        
        # Site-specific parameters
        self.elevation_m = 10.0  # Default site elevation in meters
        
        # Atmospheric conditions
        self.atmospheric_conditions = AtmosphericConditions()
        
        # Performance parameters
        self.system_efficiency = 0.85
        self.inverter_efficiency = 0.95
        self.temperature_coefficient = -0.004
        self.reference_temperature = 25.0
        
        # Thread-safe state management
        self._lock = threading.Lock()
        self._performance_stats = {
            'calculations': 0,
            'total_time': 0.0,
            'avg_time': 0.0
        }
        
        # Calculate sunrise/sunset times
        self._calculate_daylight_hours()
        
        # Pre-compute panel data arrays for vectorization
        self._prepare_panel_arrays()
        
        print(f"ThreadSafePowerSimulator initialized:")
        print(f"  Location: {latitude:.3f}°N, {longitude:.3f}°E")
        print(f"  Panels: {len(panel_df) if panel_df is not None else 0}")
        print(f"  Daylight: {self.sunrise_hour:.2f}h - {self.sunset_hour:.2f}h")
        print(f"  System efficiency: {self.system_efficiency:.1%}")
    
    def _calculate_daylight_hours(self):
        """Calculate sunrise and sunset times for the location."""
        # Simplified calculation for tropical locations
        day_length = 12.0  # Hours of daylight
        solar_noon = 12.0
        
        self.sunrise_hour = solar_noon - day_length / 2
        self.sunset_hour = solar_noon + day_length / 2
        
        # Adjust for tropical locations (less seasonal variation)
        if abs(self.latitude) < 23.5:
            seasonal_variation = 0.5  # Hours
            day_of_year = datetime.now().timetuple().tm_yday
            seasonal_offset = seasonal_variation * math.sin(2 * math.pi * (day_of_year - 80) / 365)
            
            self.sunrise_hour += seasonal_offset
            self.sunset_hour -= seasonal_offset
    
    def _prepare_panel_arrays(self):
        """Pre-compute panel data arrays for vectorized operations."""
        if self.panel_df is None or len(self.panel_df) == 0:
            self.panel_ids = np.array([], dtype='<U10')
            self.panel_capacities = np.array([], dtype=np.float32)
            self.panel_positions = np.array([], dtype=np.float32).reshape(0, 2)
            return
        
        # Extract arrays for vectorization
        self.panel_ids = self.panel_df['panel_id'].values
        self.panel_capacities = self.panel_df.get('power_capacity', 5.0).values.astype(np.float32)
        
        # Panel positions for potential future use
        if 'x_km' in self.panel_df.columns and 'y_km' in self.panel_df.columns:
            self.panel_positions = self.panel_df[['x_km', 'y_km']].values.astype(np.float32)
        else:
            self.panel_positions = np.zeros((len(self.panel_df), 2), dtype=np.float32)
    
    def calculate_power(self, panel_df=None, coverage=None, sim_time=None, 
                       hour: Optional[float] = None, panel_coverage: Optional[Dict] = None,
                       ambient_temp: float = 30.0) -> Dict[str, Any]:
        """
        Thread-safe power calculation with vectorized operations.
        Supports both new threading interface and legacy interface.
        
        NEW INTERFACE (for PhysicsWorker):
        Args:
            panel_df: DataFrame with panel information (can be None if set in init)
            coverage: Dictionary mapping panel_id to coverage (0-1)
            sim_time: Current simulation time (seconds or hour)
            
        LEGACY INTERFACE (backward compatibility):
        Args:
            hour: Current hour of day (0-24)
            panel_coverage: Dictionary mapping panel_id to cloud coverage (0-1)
            ambient_temp: Ambient temperature in °C
            
        Returns:
            Dictionary with comprehensive power output information
        """
        start_time = time.perf_counter()
        
        try:
            # Handle parameter resolution for dual interface support
            current_hour, panel_coverage_dict, panels_df = self._resolve_parameters(
                panel_df, coverage, sim_time, hour, panel_coverage
            )
            
            if panels_df is None or len(panels_df) == 0:
                return self._get_zero_power_result()
            
            # Check if it's nighttime
            if current_hour < self.sunrise_hour or current_hour > self.sunset_hour:
                return self._get_nighttime_result(panels_df)
            
            # Calculate solar position
            day_of_year = datetime.now().timetuple().tm_yday
            solar_position = _cached_solar_position(
                current_hour, day_of_year, self.latitude, self.longitude, self.timezone_offset
            )
            
            if solar_position["elevation"] <= 0:
                return self._get_nighttime_result(panels_df)
            
            # Vectorized irradiance calculation
            ghi_array = _vectorized_irradiance_calculation(
                np.array([solar_position["elevation"]], dtype=np.float32),
                np.array([solar_position["air_mass"]], dtype=np.float32),
                day_of_year,
                self.atmospheric_conditions
            )
            
            clear_ghi = ghi_array[0] if len(ghi_array) > 0 else 0.0
            
            # Calculate plane-of-array irradiance
            clear_poa_irradiance = self._calculate_poa_irradiance(clear_ghi, solar_position)
            
            # Prepare arrays for vectorized power calculation
            n_panels = len(panels_df)
            poa_irradiances = np.full(n_panels, clear_poa_irradiance, dtype=np.float32)
            panel_capacities = panels_df.get('power_capacity', 5.0).values.astype(np.float32)
            
            # Extract coverage values for all panels
            coverages = np.zeros(n_panels, dtype=np.float32)
            panel_ids = panels_df['panel_id'].values
            
            for i, panel_id in enumerate(panel_ids):
                coverages[i] = panel_coverage_dict.get(panel_id, 0.0)
            
            # Vectorized power calculation (GIL-releasing)
            dc_powers, ac_powers = _vectorized_power_calculation(
                poa_irradiances, panel_capacities, coverages,
                self.system_efficiency, self.temperature_coefficient,
                self.reference_temperature, ambient_temp
            )
            
            # Calculate baseline powers (no cloud coverage)
            baseline_dc_powers, baseline_ac_powers = _vectorized_power_calculation(
                poa_irradiances, panel_capacities, np.zeros_like(coverages),
                self.system_efficiency, self.temperature_coefficient,
                self.reference_temperature, ambient_temp
            )
            
            # Build comprehensive result dictionary
            result = self._build_result_dict(
                panels_df, panel_ids, dc_powers, ac_powers, 
                baseline_dc_powers, baseline_ac_powers, coverages,
                solar_position, clear_poa_irradiance, ambient_temp
            )
            
            # Update performance statistics
            with self._lock:
                calculation_time = time.perf_counter() - start_time
                self._performance_stats['calculations'] += 1
                self._performance_stats['total_time'] += calculation_time
                self._performance_stats['avg_time'] = (
                    self._performance_stats['total_time'] / self._performance_stats['calculations']
                )
            
            return result
            
        except Exception as e:
            print(f"Error in power calculation: {e}")
            return self._get_zero_power_result()
    
    def _resolve_parameters(self, panel_df, coverage, sim_time, hour, panel_coverage):
        """Resolve parameters for dual interface support."""
        # Determine which interface is being used
        if panel_df is not None or coverage is not None or sim_time is not None:
            # New threading interface
            panels_df = panel_df if panel_df is not None else self.panel_df
            panel_coverage_dict = coverage or {}
            
            if sim_time is not None:
                if sim_time < 100:  # Assume it's hours if < 100
                    current_hour = float(sim_time)
                else:  # Assume it's seconds/timestamp
                    current_hour = (float(sim_time) / 3600) % 24
            else:
                current_hour = 12.0  # Default noon
        else:
            # Legacy interface
            panels_df = self.panel_df
            panel_coverage_dict = panel_coverage or {}
            current_hour = hour if hour is not None else 12.0
        
        return current_hour, panel_coverage_dict, panels_df
    
    def _calculate_poa_irradiance(self, ghi: float, solar_position: Dict[str, float]) -> float:
        """Calculate plane-of-array irradiance from global horizontal irradiance."""
        elevation_rad = math.radians(solar_position["elevation"])
        azimuth_rad = math.radians(solar_position["azimuth"])
        
        if solar_position["elevation"] <= 0:
            return 0.0
        
        # Simplified POA calculation for horizontal panels
        # For tilted panels, would need more complex geometry
        if abs(self.panel_tilt) < 0.01:  # Horizontal panels
            return ghi
        else:
            # Simplified calculation for tilted panels
            cos_incidence = (math.sin(elevation_rad) * math.cos(self.panel_tilt) + 
                            math.cos(elevation_rad) * math.sin(self.panel_tilt) * 
                            math.cos(azimuth_rad - self.panel_azimuth))
            cos_incidence = max(0, cos_incidence)
            
            # Assume 70% of GHI is direct normal irradiance
            dni_estimate = ghi * 0.7 / math.sin(elevation_rad) if math.sin(elevation_rad) > 0 else 0
            dhi_estimate = ghi * 0.3  # Assume 30% diffuse
            
            direct_poa = dni_estimate * cos_incidence
            diffuse_poa = dhi_estimate * (1 + math.cos(self.panel_tilt)) / 2
            
            return max(0, direct_poa + diffuse_poa)
    
    def _build_result_dict(self, panels_df, panel_ids, dc_powers, ac_powers,
                          baseline_dc_powers, baseline_ac_powers, coverages,
                          solar_position, poa_irradiance, ambient_temp) -> Dict[str, Any]:
        """Build comprehensive result dictionary."""
        # Calculate totals
        total_dc_power = float(np.sum(dc_powers))
        total_ac_power = float(np.sum(ac_powers))
        total_baseline_dc = float(np.sum(baseline_dc_powers))
        total_baseline_ac = float(np.sum(baseline_ac_powers))
        
        # Calculate farm-level statistics
        farm_reduction_pct = ((total_baseline_dc - total_dc_power) / total_baseline_dc * 100 
                             if total_baseline_dc > 0 else 0.0)
        
        performance_ratio = (total_ac_power / total_baseline_ac 
                           if total_baseline_ac > 0 else 0.0)
        
        # Build panel details dictionary
        panel_details = {}
        for i, panel_id in enumerate(panel_ids):
            panel_details[panel_id] = {
                "baseline": float(baseline_dc_powers[i]),
                "final_power": float(dc_powers[i]),
                "ac_power": float(ac_powers[i]),
                "baseline_ac": float(baseline_ac_powers[i]),
                "coverage": float(coverages[i]),
                "capacity": float(panels_df.iloc[i].get('power_capacity', 5.0)),
                "poa_irradiance": poa_irradiance,
                "cell_temperature": ambient_temp + 15,  # Simplified cell temp
                "efficiency": float(ac_powers[i] / panels_df.iloc[i].get('power_capacity', 5.0)) if panels_df.iloc[i].get('power_capacity', 5.0) > 0 else 0
            }
        
        # Create comprehensive result
        result = panel_details.copy()
        result.update({
            # Power totals
            "total": total_dc_power,
            "total_ac": total_ac_power,
            "baseline_total": total_baseline_dc,
            "baseline_total_ac": total_baseline_ac,
            
            # Performance metrics
            "farm_reduction_pct": farm_reduction_pct,
            "performance_ratio": performance_ratio,
            "capacity_factor": total_ac_power / (len(panels_df) * 5.0) if len(panels_df) > 0 else 0,
            
            # Environmental conditions
            "poa_irradiance": poa_irradiance,
            "ambient_temperature": ambient_temp,
            "solar_elevation": solar_position["elevation"],
            "solar_azimuth": solar_position["azimuth"],
            
            # System information
            "panels_count": len(panels_df),
            "inverter_efficiency": self.inverter_efficiency,
            "system_efficiency": self.system_efficiency,
            
            # Simplified irradiance components
            "dni": poa_irradiance * 0.7,  # Estimate
            "dhi": poa_irradiance * 0.3,  # Estimate
            "ghi": poa_irradiance
        })
        
        return result
    
    def _get_zero_power_result(self) -> Dict[str, Any]:
        """Return zero power result when no panels are configured."""
        return {
            "total": 0.0,
            "total_ac": 0.0,
            "baseline_total": 0.0,
            "baseline_total_ac": 0.0,
            "farm_reduction_pct": 0.0,
            "performance_ratio": 0.0,
            "capacity_factor": 0.0,
            "panels_count": 0,
            "poa_irradiance": 0.0,
            "solar_elevation": 0.0,
            "solar_azimuth": 0.0,
            "dni": 0.0,
            "dhi": 0.0,
            "ghi": 0.0
        }
    
    def _get_nighttime_result(self, panels_df) -> Dict[str, Any]:
        """Return nighttime result with zero power generation."""
        panels_count = len(panels_df) if panels_df is not None else 0
        
        result = {
            "total": 0.0,
            "total_ac": 0.0,
            "baseline_total": 0.0,
            "baseline_total_ac": 0.0,
            "farm_reduction_pct": 0.0,
            "performance_ratio": 0.0,
            "capacity_factor": 0.0,
            "panels_count": panels_count,
            "poa_irradiance": 0.0,
            "solar_elevation": 0.0,
            "solar_azimuth": 0.0,
            "dni": 0.0,
            "dhi": 0.0,
            "ghi": 0.0
        }
        
        # Add individual panel data for nighttime
        if panels_df is not None:
            for _, row in panels_df.iterrows():
                panel_id = row["panel_id"]
                capacity = row.get("power_capacity", 5.0)
                
                result[panel_id] = {
                    "baseline": 0.0,
                    "final_power": 0.0,
                    "ac_power": 0.0,
                    "baseline_ac": 0.0,
                    "coverage": 0.0,
                    "capacity": capacity,
                    "poa_irradiance": 0.0,
                    "cell_temperature": 20.0,  # Nighttime temperature
                    "efficiency": 0.0
                }
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get thread-safe performance statistics."""
        with self._lock:
            return self._performance_stats.copy()
    
    def clear_cache(self):
        """Clear cached calculations."""
        _cached_solar_position.cache_clear()
    
    # Legacy interface support for backward compatibility
    def calculate_solar_position(self, hour: float, day_of_year: int = None) -> Dict[str, float]:
        """Legacy method for solar position calculation."""
        if day_of_year is None:
            day_of_year = datetime.now().timetuple().tm_yday
        return _cached_solar_position(hour, day_of_year, self.latitude, self.longitude, self.timezone_offset)
    
    def step(self, *args, **kwargs):
        """Legacy method - redirects to calculate_power for backward compatibility."""
        return self.calculate_power(*args, **kwargs)
    
    def simulate_power(self, *args, **kwargs):
        """Alternative method name - redirects to calculate_power."""
        return self.calculate_power(*args, **kwargs)
    
    def compute_output(self, *args, **kwargs):
        """Alternative method name - redirects to calculate_power."""
        return self.calculate_power(*args, **kwargs)

# Backward compatibility alias
PowerSimulator = ThreadSafePowerSimulator

# Factory functions for easy creation
def create_power_simulator(panel_df, location_config=None):
    """
    Factory function to create a thread-safe PowerSimulator.
    
    Args:
        panel_df: Panel DataFrame
        location_config: Dictionary with location parameters
        
    Returns:
        Configured ThreadSafePowerSimulator instance
    """
    if location_config is None:
        # Default to Sri Lanka (Negombo)
        location_config = {
            "latitude": 6.9271,
            "longitude": 79.8612,
            "timezone_offset": 5.5,
            "elevation_m": 10.0
        }
    
    return ThreadSafePowerSimulator(
        panel_df=panel_df,
        latitude=location_config.get("latitude", 6.9271),
        longitude=location_config.get("longitude", 79.8612),
        timezone_offset=location_config.get("timezone_offset", 5.5)
    )

def create_tropical_power_simulator(panel_df):
    """Create a PowerSimulator optimized for tropical locations."""
    simulator = ThreadSafePowerSimulator(
        panel_df=panel_df,
        latitude=6.9271,  # Tropical latitude
        longitude=79.8612,
        timezone_offset=5.5
    )
    
    # Adjust for tropical conditions
    simulator.atmospheric_conditions.turbidity = 3.0  # Higher turbidity
    simulator.atmospheric_conditions.water_vapor = 3.0  # Higher humidity
    simulator.temperature_coefficient = -0.0035  # Better temp coefficient for tropical modules
    
    return simulator

# Performance testing function
def benchmark_power_simulator():
    """Benchmark the power simulator for threading performance."""
    import pandas as pd
    import time
    
    print("Benchmarking ThreadSafePowerSimulator...")
    
    # Create test data
    panel_data = [
        {'panel_id': f'P{i:03d}', 'x_km': i * 2.0, 'y_km': 10.0, 'power_capacity': 5.0}
        for i in range(100)
    ]
    panel_df = pd.DataFrame(panel_data)
    
    simulator = create_power_simulator(panel_df)
    
    # Test coverage data
    test_coverage = {f'P{i:03d}': 0.3 if i % 3 == 0 else 0.0 for i in range(100)}
    
    # Benchmark new interface
    print("Testing new interface (calculate_power with panel_df, coverage, sim_time)...")
    start_time = time.time()
    iterations = 100
    
    for i in range(iterations):
        result = simulator.calculate_power(panel_df, test_coverage, 12.0)  # Noon
    
    end_time = time.time()
    avg_time_ms = (end_time - start_time) / iterations * 1000
    
    print(f"New interface - Average: {avg_time_ms:.2f} ms")
    
    # Benchmark legacy interface
    print("Testing legacy interface (calculate_power with hour, panel_coverage)...")
    start_time = time.time()
    
    for i in range(iterations):
        result = simulator.calculate_power(hour=12.0, panel_coverage=test_coverage)
    
    end_time = time.time()
    avg_time_ms = (end_time - start_time) / iterations * 1000
    
    print(f"Legacy interface - Average: {avg_time_ms:.2f} ms")
    
    # Test threading safety
    print("Testing thread safety...")
    import threading
    
    def worker():
        for _ in range(10):
            simulator.calculate_power(panel_df, test_coverage, 12.0)
    
    threads = [threading.Thread(target=worker) for _ in range(4)]
    
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end_time = time.time()
    
    print(f"Multi-thread test completed in {(end_time - start_time)*1000:.2f} ms")
    print("Benchmark completed successfully!")

# Testing and validation functions
def test_power_simulator():
    """Test the PowerSimulator with sample data."""
    import pandas as pd
    
    sample_panels = pd.DataFrame({
        'panel_id': [f'P{i:03d}' for i in range(1, 11)],
        'x_km': [i * 2 for i in range(10)],
        'y_km': [5.0] * 10,
        'power_capacity': [5.0] * 10
    })
    
    # Create simulator
    simulator = create_power_simulator(sample_panels)
    
    # Test both interfaces
    test_hours = [6, 9, 12, 15, 18, 21]
    test_coverage = {'P001': 0.5, 'P002': 0.3}
    
    print("Power Simulator Test Results:")
    print("=" * 50)
    
    for hour in test_hours:
        # Test new interface
        result_new = simulator.calculate_power(sample_panels, test_coverage, hour)
        
        # Test legacy interface
        result_legacy = simulator.calculate_power(hour=hour, panel_coverage=test_coverage)
        
        print(f"Hour {hour:2d}:00 - "
              f"Total: {result_new['total']:6.2f} kW, "
              f"AC: {result_new['total_ac']:6.2f} kW, "
              f"Solar Elev: {result_new['solar_elevation']:5.1f}°, "
              f"Reduction: {result_new['farm_reduction_pct']:4.1f}%")
    
    print("=" * 50)
    print("Test completed successfully!")

if __name__ == "__main__":
    # Run tests when module is executed directly
    test_power_simulator()
    benchmark_power_simulator()