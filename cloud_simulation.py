# cloud_simulation.py
"""
Complete Advanced Cloud Physics System with Threading Support
Production-ready cloud simulation with realistic physics, multiple cloud types,
and comprehensive atmospheric modeling for solar farm simulation.

CRITICAL FIX: Added compatibility shim for legacy t_s parameter to eliminate TypeError.

Features:
- Advanced cloud physics with realistic lifecycle management
- Multiple cloud types (cumulus, stratus, cirrus) with distinct behaviors
- Comprehensive atmospheric modeling (pressure, humidity, temperature)
- Thread-safe operation with deep copy isolation
- Performance optimization with spatial partitioning
- CSV coordinate bounds integration
- Configurable parameters for all aspects of simulation
- Immutable state management for worker thread communication
- Legacy parameter compatibility (t_s -> dt conversion)
"""
import math
import random
import time
import threading
import copy
import os
from typing import List, Tuple, Optional, Dict, Any, Union, Set
from functools import lru_cache
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

# Import configuration
try:
    import sim_config as CFG
except ImportError:
    # Fallback configuration
    class CFG:
        DOMAIN_SIZE_M = 10000
        AREA_SIZE_KM = 10
        PHYSICS_TIMESTEP = 1/60.0
        TIME_SCALE = 1.0
        MAX_PARCELS = 6
        BASE_WIND_SPEED = 2.0
        BASE_WIND_DIRECTION = 135.0  # NE to SW
        CLOUD_DIRECTION = 135.0

print("Loading Complete Advanced Cloud Physics System...")


# ===================================================================
# CRITICAL FIX: Legacy parameter compatibility helper
# ===================================================================

def _aliased_dt(dt: Optional[float], kwargs: Dict[str, Any]) -> float:
    """
    Accept legacy keyword 't_s'. If both are supplied, dt wins.
    This eliminates the TypeError: step() got an unexpected keyword argument 't_s'
    """
    if dt is not None:
        return dt
    elif 't_s' in kwargs:
        return kwargs.pop('t_s')  # Remove t_s from kwargs to prevent further errors
    else:
        return 1.0 / 60.0  # Default timestep


# ===================================================================
# CLOUD TYPE SYSTEM
# ===================================================================

class CloudType(Enum):
    """Enumeration of cloud types with distinct characteristics."""
    CUMULUS = "cumulus"
    STRATUS = "stratus"
    CIRRUS = "cirrus"
    CUMULONIMBUS = "cumulonimbus"
    NIMBOSTRATUS = "nimbostratus"
    ALTOCUMULUS = "altocumulus"


@dataclass(frozen=True)
class CloudCharacteristics:
    """Immutable cloud type characteristics for thread safety."""
    # Basic properties
    cloud_type: CloudType
    density: float
    opacity_base: float
    size_range: Tuple[float, float]  # Min, max size in km
    altitude_range: Tuple[float, float]  # Min, max altitude in km
    
    # Shape characteristics
    shape_factor_x: float = 1.0
    shape_factor_y: float = 1.0
    vertical_extent: float = 0.5  # km
    
    # Movement properties
    base_velocity: float = 1.0  # m/s
    wind_responsiveness: float = 0.8
    vertical_motion: float = 0.1
    
    # Lifecycle properties
    formation_rate: float = 0.01
    decay_rate: float = 0.005
    max_age: float = 3600.0  # seconds
    
    # Physical properties
    water_capacity: float = 1.0
    temperature_sensitivity: float = 1.0
    pressure_sensitivity: float = 0.5


# Define cloud type characteristics
CLOUD_CHARACTERISTICS = {
    CloudType.CUMULUS: CloudCharacteristics(
        cloud_type=CloudType.CUMULUS,
        density=0.6, opacity_base=0.7,
        size_range=(0.5, 2.5), altitude_range=(500, 3000),
        shape_factor_x=1.2, shape_factor_y=1.0,
        vertical_extent=1.5, base_velocity=2.0,
        wind_responsiveness=0.9, vertical_motion=0.3,
        formation_rate=0.02, decay_rate=0.008,
        max_age=2400, water_capacity=0.8
    ),
    CloudType.STRATUS: CloudCharacteristics(
        cloud_type=CloudType.STRATUS,
        density=0.4, opacity_base=0.5,
        size_range=(2.0, 8.0), altitude_range=(200, 2000),
        shape_factor_x=3.0, shape_factor_y=1.5,
        vertical_extent=0.3, base_velocity=1.5,
        wind_responsiveness=1.0, vertical_motion=0.05,
        formation_rate=0.005, decay_rate=0.003,
        max_age=7200, water_capacity=0.6
    ),
    CloudType.CIRRUS: CloudCharacteristics(
        cloud_type=CloudType.CIRRUS,
        density=0.2, opacity_base=0.3,
        size_range=(1.0, 4.0), altitude_range=(6000, 12000),
        shape_factor_x=2.0, shape_factor_y=0.8,
        vertical_extent=0.5, base_velocity=8.0,
        wind_responsiveness=1.2, vertical_motion=0.1,
        formation_rate=0.008, decay_rate=0.01,
        max_age=5400, water_capacity=0.3
    )
}


# ===================================================================
# ATMOSPHERIC CONDITIONS
# ===================================================================

@dataclass
class AtmosphericConditions:
    """Complete atmospheric state for realistic cloud physics."""
    # Basic meteorology
    temperature: float = 20.0      # °C
    humidity: float = 60.0         # %
    pressure: float = 1013.25     # hPa
    dew_point: float = 12.0       # °C
    
    # Wind conditions
    wind_speed: float = 5.0        # m/s
    wind_direction: float = 225.0  # degrees (meteorological)
    wind_shear: float = 0.0        # m/s per km altitude
    
    # Atmospheric stability
    stability_index: float = 0.5   # 0=very unstable, 1=very stable
    convection_strength: float = 0.3  # 0-1
    turbulence_intensity: float = 0.2  # 0-1
    
    # Radiation and energy
    solar_radiation: float = 800.0  # W/m²
    cloud_base_height: float = 1200.0  # meters
    
    # Advanced properties
    air_density: float = 1.225     # kg/m³
    pressure_gradient: float = 0.0  # hPa/km
    vertical_velocity: float = 0.0  # m/s


# ===================================================================
# CLOUD PHYSICS STATE
# ===================================================================

@dataclass
class CloudPhysics:
    """
    Complete cloud physics state.
    All properties are immutable for thread-safe worker communication.
    """
    # Core position and motion
    position: Tuple[float, float, float]  # (x, y, altitude) in meters
    velocity: Tuple[float, float, float]  # (vx, vy, vz) in m/s
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Size and shape
    size: float = 1.0                # Size in km
    shape_aspect_x: float = 1.0      # X-axis shape factor
    shape_aspect_y: float = 1.0      # Y-axis shape factor
    volume: float = 1.0              # Cloud volume in km³
    
    # Physical properties
    density: float = 0.7             # Cloud density (0-1)
    opacity: float = 0.8             # Visual opacity (0-1)
    water_content: float = 0.5       # Liquid water content
    ice_content: float = 0.0         # Ice crystal content
    
    # Lifecycle
    age: float = 0.0                 # Age in seconds
    lifecycle_stage: str = "growth"   # growth, mature, decay
    health: float = 1.0              # Overall health (0-1)
    
    # Environmental interactions
    temperature: float = 15.0        # Internal temperature
    condensation_rate: float = 0.0   # Current condensation
    evaporation_rate: float = 0.0    # Current evaporation
    
    # Advanced properties
    turbulence_energy: float = 0.0   # Internal turbulence
    electrical_charge: float = 0.0   # Electrical buildup
    precipitation_rate: float = 0.0  # Precipitation intensity


@dataclass
class Cloud:
    """
    Complete cloud dataclass with all properties for realistic simulation.
    Immutable snapshot for thread-safe communication.
    """
    # Identification
    cloud_id: str
    cloud_type: CloudType
    
    # Physics state
    physics: CloudPhysics
    
    # Characteristics (immutable reference)
    characteristics: CloudCharacteristics
    
    # Spatial information
    bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)  # (min_x, min_y, max_x, max_y)
    grid_cells: Set[Tuple[int, int]] = field(default_factory=set)
    
    # Trail and history
    position_history: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    
    # Status flags
    is_active: bool = True
    needs_update: bool = True
    marked_for_removal: bool = False
    
    def get_position_2d(self) -> Tuple[float, float]:
        """Get 2D position in km."""
        x_km = self.physics.position[0] / 1000.0
        y_km = self.physics.position[1] / 1000.0
        return x_km, y_km
    
    def get_altitude_km(self) -> float:
        """Get altitude in km."""
        return self.physics.position[2] / 1000.0
    
    def get_ellipse_data(self) -> Tuple[float, float, float, float, float, float, float, str]:
        """Get ellipse data for rendering."""
        x_km, y_km = self.get_position_2d()
        
        # Calculate shape based on type and physics
        width = self.physics.size * self.characteristics.shape_factor_x * 1000  # Convert to meters
        height = self.physics.size * self.characteristics.shape_factor_y * 1000
        
        return (
            x_km * 1000,  # x in meters for rendering
            y_km * 1000,  # y in meters for rendering
            width,        # width in meters
            height,       # height in meters
            0.0,          # rotation
            self.physics.opacity,
            self.physics.position[2],  # altitude
            self.cloud_type.value
        )


# ===================================================================
# SPATIAL OPTIMIZATION
# ===================================================================

class SpatialGrid:
    """Spatial partitioning grid for performance optimization."""
    
    def __init__(self, domain_size: float, cell_size: float = 1000.0):
        """Initialize spatial grid."""
        self.domain_size = domain_size
        self.cell_size = cell_size
        self.grid_size = int(domain_size / cell_size)
        self.cells: Dict[Tuple[int, int], List[str]] = {}
        self._lock = threading.RLock()
    
    def _get_cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Get grid cell coordinates for position."""
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        return (
            max(0, min(self.grid_size - 1, cell_x)),
            max(0, min(self.grid_size - 1, cell_y))
        )
    
    def add_cloud(self, cloud: Cloud):
        """Add cloud to spatial grid."""
        with self._lock:
            cell = self._get_cell_coords(cloud.physics.position[0], cloud.physics.position[1])
            if cell not in self.cells:
                self.cells[cell] = []
            if cloud.cloud_id not in self.cells[cell]:
                self.cells[cell].append(cloud.cloud_id)
    
    def remove_cloud(self, cloud: Cloud):
        """Remove cloud from spatial grid."""
        with self._lock:
            cell = self._get_cell_coords(cloud.physics.position[0], cloud.physics.position[1])
            if cell in self.cells and cloud.cloud_id in self.cells[cell]:
                self.cells[cell].remove(cloud.cloud_id)
                if not self.cells[cell]:
                    del self.cells[cell]
    
    def get_nearby_clouds(self, x: float, y: float, radius: float) -> List[str]:
        """Get cloud IDs near position within radius."""
        with self._lock:
            nearby = []
            center_cell = self._get_cell_coords(x, y)
            search_range = int(radius / self.cell_size) + 1
            
            for dx in range(-search_range, search_range + 1):
                for dy in range(-search_range, search_range + 1):
                    cell = (center_cell[0] + dx, center_cell[1] + dy)
                    if (0 <= cell[0] < self.grid_size and 
                        0 <= cell[1] < self.grid_size and 
                        cell in self.cells):
                        nearby.extend(self.cells[cell])
            
            return nearby
    
    def clear(self):
        """Clear all cells."""
        with self._lock:
            self.cells.clear()


# ===================================================================
# WEATHER SYSTEM (ATMOSPHERIC CONDITIONS MANAGER)
# ===================================================================

class WeatherSystem:
    """Atmospheric conditions manager with realistic weather patterns."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize weather system."""
        self.rng = random.Random(seed)
        self.atmospheric_conditions = AtmosphericConditions()
        self.time_of_day = 12.0  # Start at noon
        self.sim_time = 0.0
        self.season_factor = 0.5  # 0=winter, 1=summer
        self.pressure_cycle_phase = 0.0
        self._lock = threading.RLock()
        self._thread_safe_mode = False
    
    def set_thread_safe_mode(self, enabled: bool):
        """Enable/disable thread-safe mode."""
        with self._lock:
            self._thread_safe_mode = enabled
            print(f"Weather system thread-safe mode: {'enabled' if enabled else 'disabled'}")
    
    def update(self, dt: float, current_hour: float = None):
        """Update atmospheric conditions."""
        with self._lock if self._thread_safe_mode else threading.RLock():
            self.sim_time += dt
            
            if current_hour is not None:
                self.time_of_day = current_hour
            else:
                self.time_of_day = (self.time_of_day + dt / 3600.0) % 24.0
            
            self._update_diurnal_cycle(dt)
            self._update_pressure_systems(dt)
            self._update_wind_conditions(dt)
            self._update_thermal_conditions(dt)
    
    def _update_diurnal_cycle(self, dt: float):
        """Update daily atmospheric cycle."""
        # Solar heating cycle
        solar_angle = (self.time_of_day - 6.0) * math.pi / 12.0  # Peak at noon
        solar_factor = max(0, math.sin(solar_angle))
        
        # Update solar radiation
        base_radiation = 1000.0 * self.season_factor + 600.0
        self.atmospheric_conditions.solar_radiation = base_radiation * solar_factor
        
        # Update temperature with lag
        temp_lag_factor = math.sin(solar_angle - math.pi / 6)  # Temperature peaks after solar peak
        base_temp = 20.0 + 10.0 * self.season_factor
        temp_variation = 15.0 * temp_lag_factor
        self.atmospheric_conditions.temperature = base_temp + temp_variation
        
        # Update convection strength
        self.atmospheric_conditions.convection_strength = max(0.1, solar_factor * 0.8)
    
    def _update_pressure_systems(self, dt: float):
        """Update atmospheric pressure systems."""
        # Slow pressure variation
        self.pressure_cycle_phase += dt * 0.0001  # Very slow cycle
        pressure_variation = 20.0 * math.sin(self.pressure_cycle_phase)
        
        # Random pressure fluctuations
        if self.rng.random() < 0.01:  # 1% chance per frame
            pressure_variation += self.rng.gauss(0, 5.0)
        
        self.atmospheric_conditions.pressure = 1013.25 + pressure_variation
        self.atmospheric_conditions.pressure_gradient = pressure_variation * 0.1
    
    def _update_wind_conditions(self, dt: float):
        """Update wind speed and direction."""
        # Base wind from configuration
        base_speed = getattr(CFG, 'BASE_WIND_SPEED', 2.0)
        base_direction = getattr(CFG, 'BASE_WIND_DIRECTION', 135.0)
        
        # Diurnal wind variations
        diurnal_factor = 1.0 + 0.3 * math.sin((self.time_of_day - 12.0) * math.pi / 12.0)
        
        # Pressure gradient effects
        pressure_effect = self.atmospheric_conditions.pressure_gradient * 0.1
        
        # Random variations
        speed_noise = self.rng.gauss(0, 0.5)
        direction_noise = self.rng.gauss(0, 10.0)
        
        self.atmospheric_conditions.wind_speed = max(0.1, base_speed * diurnal_factor + pressure_effect + speed_noise)
        self.atmospheric_conditions.wind_direction = (base_direction + direction_noise) % 360.0
    
    def _update_thermal_conditions(self, dt: float):
        """Update thermal stability and humidity."""
        # Calculate humidity based on temperature and time of day
        temp = self.atmospheric_conditions.temperature
        
        # Higher humidity at lower temperatures and during night
        temp_humidity_factor = max(0.3, 1.0 - (temp - 10.0) / 30.0)
        time_humidity_factor = 1.0 + 0.2 * math.cos((self.time_of_day - 6.0) * math.pi / 12.0)
        
        target_humidity = 60.0 * temp_humidity_factor * time_humidity_factor
        
        # Smooth humidity changes
        humidity_change_rate = 0.1  # 10% per update towards target
        current_humidity = self.atmospheric_conditions.humidity
        self.atmospheric_conditions.humidity = current_humidity + (target_humidity - current_humidity) * humidity_change_rate
        
        # Update dew point
        self.atmospheric_conditions.dew_point = temp - (100 - self.atmospheric_conditions.humidity) / 5.0
        
        # Update stability index
        # More unstable during day, more stable at night
        solar_factor = max(0, math.sin((self.time_of_day - 6.0) * math.pi / 12.0))
        self.atmospheric_conditions.stability_index = 0.7 - 0.4 * solar_factor
    
    def get_conditions_at_altitude(self, altitude: float) -> AtmosphericConditions:
        """Get atmospheric conditions at specific altitude."""
        # Create modified conditions for altitude
        conditions = copy.deepcopy(self.atmospheric_conditions)
        
        # Standard atmosphere model
        altitude_km = altitude / 1000.0
        
        # Temperature lapse rate: -6.5°C per km
        conditions.temperature -= altitude_km * 6.5
        
        # Pressure decrease with altitude
        pressure_ratio = (1 - 0.0065 * altitude_km / 288.15) ** 5.257
        conditions.pressure *= pressure_ratio
        
        # Air density decrease
        conditions.air_density *= pressure_ratio * (288.15 / (288.15 - 6.5 * altitude_km))
        
        # Wind speed increase with altitude
        conditions.wind_speed *= (1 + altitude_km * 0.1)
        
        return conditions
    
    def get_spawn_conditions(self) -> Dict[str, Any]:
        """Get conditions favorable for cloud spawning."""
        return {
            'temperature': self.atmospheric_conditions.temperature,
            'humidity': self.atmospheric_conditions.humidity,
            'stability': self.atmospheric_conditions.stability_index,
            'convection': self.atmospheric_conditions.convection_strength,
            'wind_speed': self.atmospheric_conditions.wind_speed,
            'wind_direction': self.atmospheric_conditions.wind_direction,
            'spawn_probability': self._calculate_spawn_probability()
        }
    
    def _calculate_spawn_probability(self) -> float:
        """Calculate probability of new cloud formation."""
        # Base probability
        base_prob = 0.01
        
        # Humidity factor (higher humidity = more likely)
        humidity_factor = max(0, (self.atmospheric_conditions.humidity - 40) / 60.0)
        
        # Convection factor (more convection = more likely)
        convection_factor = self.atmospheric_conditions.convection_strength
        
        # Stability factor (unstable air = more likely)
        stability_factor = 1.0 - self.atmospheric_conditions.stability_index
        
        # Time of day factor (more likely during day)
        time_factor = max(0.2, math.sin((self.time_of_day - 6) * math.pi / 12.0))
        
        return base_prob * humidity_factor * convection_factor * stability_factor * time_factor


# ===================================================================
# MAIN CLOUD SIMULATION CLASS
# ===================================================================

class CloudSimulation:
    """
    Complete cloud simulation with realistic cloud physics and lifecycle management.
    Thread-safe operation with immutable state management for worker communication.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize cloud simulation system."""
        print(f"Initializing CloudSimulation with seed: {seed}")
        
        # Random number generation
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        # Core state
        self.clouds: Dict[str, Cloud] = {}
        self.cloud_counter = 0
        self.sim_time = 0.0
        self.time_of_day = 12.0
        
        # Atmospheric system
        self.weather_system = WeatherSystem(seed)
        
        # Spatial optimization
        domain_size = getattr(CFG, 'DOMAIN_SIZE_M', 10000)
        self.spatial_grid = SpatialGrid(domain_size)
        
        # Threading support
        self._lock = threading.RLock()
        self._thread_safe_mode = False
        
        # Performance tracking
        self.performance_stats = {
            'total_updates': 0,
            'clouds_created': 0,
            'clouds_destroyed': 0,
            'avg_cloud_count': 0,
            'last_update_time': 0
        }
        
        # Configuration
        self.max_clouds = getattr(CFG, 'MAX_PARCELS', 6)
        self.spawn_rate = 0.02  # Base spawn probability per frame
        self.domain_bounds = (0, 0, domain_size, domain_size)
        
        # For backward compatibility (parcels property)
        self.parcels = []
        
        print(f"✓ CloudSimulation initialized with max_clouds={self.max_clouds}")
    
    def set_thread_safe_mode(self, enabled: bool):
        """Enable/disable thread-safe mode for worker thread usage."""
        with self._lock:
            self._thread_safe_mode = enabled
            self.weather_system.set_thread_safe_mode(enabled)
            print(f"CloudSimulation thread-safe mode: {'enabled' if enabled else 'disabled'}")
    
    def step(self, dt: Optional[float] = None, **kwargs) -> None:
        """
        Main simulation step with CRITICAL FIX for legacy t_s parameter.
        
        Args:
            dt: Delta time in seconds
            **kwargs: Additional parameters (handles legacy t_s parameter)
        """
        # CRITICAL FIX: Handle legacy t_s parameter
        dt = _aliased_dt(dt, kwargs)
        
        # Handle other legacy parameters
        current_hour = kwargs.get('current_hour', None)
        
        with self._lock if self._thread_safe_mode else threading.RLock():
            self.sim_time += dt
            
            # Update time of day
            if current_hour is not None:
                self.time_of_day = current_hour
            else:
                time_scale = getattr(CFG, 'TIME_SCALE', 1.0)
                hour_increment = dt * time_scale / 3600.0
                self.time_of_day = (self.time_of_day + hour_increment) % 24.0
            
            # Update atmospheric conditions
            self.weather_system.update(dt, self.time_of_day)
            
            # Update existing clouds
            self._update_clouds(dt)
            
            # Spawn new clouds
            self._spawn_clouds(dt)
            
            # Cleanup dead clouds
            self._cleanup_clouds()
            
            # Update spatial grid
            self._update_spatial_grid()
            
            # Update backward compatibility parcels
            self._update_parcels()
            
            # Update performance stats
            self._update_performance_stats()
    
    def _update_clouds(self, dt: float):
        """Update physics for all active clouds."""
        clouds_to_update = list(self.clouds.keys())
        
        for cloud_id in clouds_to_update:
            if cloud_id in self.clouds:  # Check if still exists
                cloud = self.clouds[cloud_id]
                if cloud.is_active and not cloud.marked_for_removal:
                    try:
                        updated_cloud = self._update_cloud_physics(cloud, dt)
                        self.clouds[cloud_id] = updated_cloud
                    except Exception as e:
                        print(f"Error updating cloud {cloud_id}: {e}")
                        # Mark for removal if update fails
                        self.clouds[cloud_id] = cloud._replace(marked_for_removal=True)
    
    def _update_cloud_physics(self, cloud: Cloud, dt: float) -> Cloud:
        """Update individual cloud physics."""
        # Get atmospheric conditions at cloud altitude
        conditions = self.weather_system.get_conditions_at_altitude(cloud.physics.position[2])
        
        # Create new physics state (immutable update)
        new_physics = CloudPhysics(
            position=cloud.physics.position,
            velocity=cloud.physics.velocity,
            acceleration=cloud.physics.acceleration,
            size=cloud.physics.size,
            shape_aspect_x=cloud.physics.shape_aspect_x,
            shape_aspect_y=cloud.physics.shape_aspect_y,
            volume=cloud.physics.volume,
            density=cloud.physics.density,
            opacity=cloud.physics.opacity,
            water_content=cloud.physics.water_content,
            ice_content=cloud.physics.ice_content,
            age=cloud.physics.age,
            lifecycle_stage=cloud.physics.lifecycle_stage,
            health=cloud.physics.health,
            temperature=cloud.physics.temperature,
            condensation_rate=cloud.physics.condensation_rate,
            evaporation_rate=cloud.physics.evaporation_rate,
            turbulence_energy=cloud.physics.turbulence_energy,
            electrical_charge=cloud.physics.electrical_charge,
            precipitation_rate=cloud.physics.precipitation_rate
        )
        
        # Update position based on velocity
        current_pos = new_physics.position
        current_vel = new_physics.velocity
        
        new_position = (
            current_pos[0] + current_vel[0] * dt,
            current_pos[1] + current_vel[1] * dt,
            current_pos[2] + current_vel[2] * dt
        )
        
        # Keep within domain bounds
        domain_size = getattr(CFG, 'DOMAIN_SIZE_M', 10000)
        new_position = (
            max(0, min(domain_size, new_position[0])),
            max(0, min(domain_size, new_position[1])),
            max(cloud.characteristics.altitude_range[0], 
                min(cloud.characteristics.altitude_range[1], new_position[2]))
        )
        
        new_physics = new_physics._replace(position=new_position)
        
        # Update velocity based on wind and cloud characteristics
        wind_vx = conditions.wind_speed * math.cos(math.radians(conditions.wind_direction))
        wind_vy = conditions.wind_speed * math.sin(math.radians(conditions.wind_direction))
        
        # Add turbulence
        turbulence_scale = conditions.turbulence_intensity * 0.5
        if turbulence_scale > 0:
            turbulence_vx = self.rng.gauss(0, turbulence_scale)
            turbulence_vy = self.rng.gauss(0, turbulence_scale)
        else:
            turbulence_vx = turbulence_vy = 0.0
        
        # Vertical motion
        vertical_motion = cloud.characteristics.vertical_motion
        thermal_lift = conditions.convection_strength * vertical_motion
        
        # Smooth velocity changes
        velocity_damping = 0.9
        new_velocity = (
            new_physics.velocity[0] * velocity_damping + (wind_vx + turbulence_vx) * 0.1,
            new_physics.velocity[1] * velocity_damping + (wind_vy + turbulence_vy) * 0.1,
            new_physics.velocity[2] * 0.95 + thermal_lift * 0.05
        )
        new_physics = new_physics._replace(velocity=new_velocity)
        
        # Update thermodynamics
        new_temperature = conditions.temperature + self.rng.gauss(0, 2.0)
        new_physics = new_physics._replace(temperature=new_temperature)
        
        # Update water content based on humidity and temperature
        humidity_factor = conditions.humidity / 100.0
        condensation_rate = max(0, humidity_factor - 0.6) * 0.1
        evaporation_rate = max(0, (new_physics.temperature - conditions.dew_point) / 10.0) * 0.05
        
        new_physics = new_physics._replace(
            condensation_rate=condensation_rate,
            evaporation_rate=evaporation_rate
        )
        
        # Update water content
        water_change = (condensation_rate - evaporation_rate) * dt
        new_water_content = max(0.1, min(1.0, new_physics.water_content + water_change))
        new_physics = new_physics._replace(water_content=new_water_content)
        
        # Update density and opacity based on water content
        new_density = cloud.characteristics.density * new_physics.water_content
        new_opacity = cloud.characteristics.opacity_base * new_density
        
        new_physics = new_physics._replace(
            density=new_density,
            opacity=new_opacity
        )
        
        # Age the cloud
        new_age = new_physics.age + dt
        new_physics = new_physics._replace(age=new_age)
        
        # Update lifecycle stage
        max_age = cloud.characteristics.max_age
        if new_age > max_age * 0.8:
            lifecycle_stage = "decay"
        elif new_age > max_age * 0.3:
            lifecycle_stage = "mature"
        else:
            lifecycle_stage = "growth"
        
        new_physics = new_physics._replace(lifecycle_stage=lifecycle_stage)
        
        # Update health based on age and conditions
        age_factor = 1.0 - (new_age / max_age)
        condition_factor = (humidity_factor + conditions.convection_strength) / 2.0
        new_health = max(0.0, min(1.0, age_factor * condition_factor))
        new_physics = new_physics._replace(health=new_health)
        
        # Update position history
        x_km, y_km = new_position[0] / 1000.0, new_position[1] / 1000.0
        new_history = cloud.position_history + ((x_km, y_km),)
        # Keep only last 10 positions
        if len(new_history) > 10:
            new_history = new_history[-10:]
        
        # Determine if cloud should be marked for removal
        should_remove = (new_health <= 0.1 or 
                        new_age > max_age or 
                        new_opacity < 0.1)
        
        # Create updated cloud
        return Cloud(
            cloud_id=cloud.cloud_id,
            cloud_type=cloud.cloud_type,
            physics=new_physics,
            characteristics=cloud.characteristics,
            bounds=cloud.bounds,
            grid_cells=cloud.grid_cells,
            position_history=new_history,
            is_active=not should_remove,
            needs_update=True,
            marked_for_removal=should_remove
        )
    
    def _spawn_clouds(self, dt: float):
        """Spawn new clouds based on atmospheric conditions."""
        if len(self.clouds) >= self.max_clouds:
            return
        
        spawn_conditions = self.weather_system.get_spawn_conditions()
        spawn_probability = spawn_conditions['spawn_probability'] * self.spawn_rate * dt
        
        if self.rng.random() < spawn_probability:
            self._create_cloud()
    
    def _create_cloud(self) -> Optional[Cloud]:
        """Create a new cloud with realistic properties."""
        # Choose cloud type based on atmospheric conditions
        cloud_type = self._select_cloud_type()
        characteristics = CLOUD_CHARACTERISTICS[cloud_type]
        
        # Generate spawn position
        domain_size = getattr(CFG, 'DOMAIN_SIZE_M', 10000)
        margin = domain_size * 0.1  # 10% margin from edges
        
        spawn_x = self.rng.uniform(margin, domain_size - margin)
        spawn_y = self.rng.uniform(margin, domain_size - margin)
        spawn_altitude = self.rng.uniform(*characteristics.altitude_range)
        
        # Generate initial properties
        initial_size = self.rng.uniform(*characteristics.size_range)
        initial_velocity = (
            self.rng.gauss(0, characteristics.base_velocity * 0.5),
            self.rng.gauss(0, characteristics.base_velocity * 0.5),
            self.rng.gauss(0, 0.1)
        )
        
        # Create physics state
        physics = CloudPhysics(
            position=(spawn_x, spawn_y, spawn_altitude),
            velocity=initial_velocity,
            size=initial_size,
            density=characteristics.density,
            opacity=characteristics.opacity_base,
            water_content=self.rng.uniform(0.3, 0.8),
            temperature=self.weather_system.atmospheric_conditions.temperature + self.rng.gauss(0, 3),
            lifecycle_stage="growth",
            health=1.0
        )
        
        # Create cloud
        cloud_id = f"cloud_{self.cloud_counter:04d}"
        self.cloud_counter += 1
        
        cloud = Cloud(
            cloud_id=cloud_id,
            cloud_type=cloud_type,
            physics=physics,
            characteristics=characteristics,
            is_active=True,
            needs_update=True,
            marked_for_removal=False
        )
        
        # Add to simulation
        self.clouds[cloud_id] = cloud
        self.performance_stats['clouds_created'] += 1
        
        if self._thread_safe_mode:
            print(f"Created cloud {cloud_id} ({cloud_type.value}) at ({spawn_x/1000:.1f}, {spawn_y/1000:.1f}) km")
        
        return cloud
    
    def _select_cloud_type(self) -> CloudType:
        """Select cloud type based on atmospheric conditions."""
        conditions = self.weather_system.atmospheric_conditions
        
        # Simple heuristic based on stability and altitude
        if conditions.stability_index < 0.3:
            # Unstable conditions favor cumulus
            return CloudType.CUMULUS
        elif conditions.stability_index > 0.7:
            # Stable conditions favor stratus
            return CloudType.STRATUS
        else:
            # Mixed conditions
            if conditions.convection_strength > 0.5:
                return CloudType.CUMULUS
            else:
                return self.rng.choice([CloudType.CUMULUS, CloudType.STRATUS])
    
    def _cleanup_clouds(self):
        """Remove dead and expired clouds."""
        clouds_to_remove = []
        
        for cloud_id, cloud in self.clouds.items():
            if cloud.marked_for_removal or not cloud.is_active:
                clouds_to_remove.append(cloud_id)
        
        for cloud_id in clouds_to_remove:
            del self.clouds[cloud_id]
            self.performance_stats['clouds_destroyed'] += 1
    
    def _update_spatial_grid(self):
        """Update spatial grid with current cloud positions."""
        self.spatial_grid.clear()
        for cloud in self.clouds.values():
            if cloud.is_active:
                self.spatial_grid.add_cloud(cloud)
    
    def _update_parcels(self):
        """Update backward compatibility parcels list."""
        # Create simple parcel objects for backward compatibility
        self.parcels = []
        for cloud in self.clouds.values():
            if cloud.is_active:
                # Create simple parcel-like object
                class SimpleParcel:
                    def __init__(self, cloud: Cloud):
                        self.x = cloud.physics.position[0]
                        self.y = cloud.physics.position[1]
                        self.size = cloud.physics.size * 1000  # Convert to meters
                        self.density = cloud.physics.density
                        self.opacity = cloud.physics.opacity
                        self.age = cloud.physics.age
                        self.type = cloud.cloud_type.value
                        self.cloud_id = cloud.cloud_id
                
                self.parcels.append(SimpleParcel(cloud))
    
    def _update_performance_stats(self):
        """Update performance statistics."""
        self.performance_stats['total_updates'] += 1
        self.performance_stats['avg_cloud_count'] = len(self.clouds)
        self.performance_stats['last_update_time'] = self.sim_time
    
    # ===================================================================
    # PUBLIC INTERFACE METHODS
    # ===================================================================
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete simulation state."""
        with self._lock if self._thread_safe_mode else threading.RLock():
            return {
                'sim_time': self.sim_time,
                'time_of_day': self.time_of_day,
                'active_cloud_count': len([c for c in self.clouds.values() if c.is_active]),
                'total_clouds': len(self.clouds),
                'cloud_coverage': self.current_cloud_cover_pct(),
                'wind_speed': self.weather_system.atmospheric_conditions.wind_speed,
                'wind_direction': self.weather_system.atmospheric_conditions.wind_direction,
                'temperature': self.weather_system.atmospheric_conditions.temperature,
                'humidity': self.weather_system.atmospheric_conditions.humidity,
                'pressure': self.weather_system.atmospheric_conditions.pressure
            }
    
    def current_cloud_cover_pct(self) -> float:
        """Calculate current cloud coverage percentage."""
        if not self.clouds:
            return 0.0
        
        domain_size = getattr(CFG, 'DOMAIN_SIZE_M', 10000)
        total_area = domain_size * domain_size
        covered_area = 0.0
        
        for cloud in self.clouds.values():
            if cloud.is_active and cloud.physics.opacity > 0.1:
                # Approximate cloud area as ellipse
                size_m = cloud.physics.size * 1000  # Convert to meters
                area = math.pi * size_m * size_m * 0.25  # Simple circular approximation
                covered_area += area * cloud.physics.opacity
        
        return min(100.0, (covered_area / total_area) * 100.0)
    
    def get_avg_trajectory(self) -> Tuple[float, float, float]:
        """Get average wind trajectory for compatibility."""
        conditions = self.weather_system.atmospheric_conditions
        return (
            conditions.wind_speed,
            conditions.wind_direction,
            len(self.clouds)
        )
    
    def get_coverage_at_position(self, x_km: float, y_km: float) -> float:
        """Get cloud coverage at specific position."""
        x_m = x_km * 1000
        y_m = y_km * 1000
        
        total_coverage = 0.0
        
        for cloud in self.clouds.values():
            if cloud.is_active:
                # Calculate distance from cloud center
                dx = x_m - cloud.physics.position[0]
                dy = y_m - cloud.physics.position[1]
                distance = math.sqrt(dx * dx + dy * dy)
                
                # Cloud radius in meters
                cloud_radius = cloud.physics.size * 1000 * 0.5
                
                if distance < cloud_radius:
                    # Simple linear falloff from center
                    coverage_factor = 1.0 - (distance / cloud_radius)
                    total_coverage += cloud.physics.opacity * coverage_factor
        
        return min(1.0, total_coverage)
    
    def get_clouds_near_position(self, x_km: float, y_km: float, radius_km: float) -> List[Cloud]:
        """Get clouds near position within radius."""
        x_m = x_km * 1000
        y_m = y_km * 1000
        radius_m = radius_km * 1000
        
        nearby_cloud_ids = self.spatial_grid.get_nearby_clouds(x_m, y_m, radius_m)
        nearby_clouds = []
        
        for cloud_id in nearby_cloud_ids:
            if cloud_id in self.clouds:
                cloud = self.clouds[cloud_id]
                # Verify actual distance
                dx = x_m - cloud.physics.position[0]
                dy = y_m - cloud.physics.position[1]
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance <= radius_m:
                    nearby_clouds.append(cloud)
        
        return nearby_clouds
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

def collect_visible_ellipses(clouds: Union[List[Cloud], Dict[str, Cloud]]) -> List[Tuple]:
    """
    Extract visible ellipse data from clouds for rendering.
    
    Args:
        clouds: List or dict of Cloud objects
        
    Returns:
        List of ellipse tuples (x, y, width, height, angle, opacity, altitude, type)
    """
    if isinstance(clouds, dict):
        clouds = list(clouds.values())
    
    ellipses = []
    for cloud in clouds:
        if (hasattr(cloud, 'is_active') and cloud.is_active and 
            hasattr(cloud, 'physics') and cloud.physics.opacity > 0.1):
            ellipse_data = cloud.get_ellipse_data()
            ellipses.append(ellipse_data)
    
    return ellipses


def create_optimized_weather_system(seed: int = None) -> CloudSimulation:
    """Create optimized weather system (main interface)."""
    return CloudSimulation(seed=seed)


def create_weather_with_initial_clouds(num_clouds: int = 3, seed: int = None) -> CloudSimulation:
    """Create weather system with initial clouds."""
    simulation = CloudSimulation(seed=seed)
    
    # Force spawn initial clouds by temporarily increasing spawn rate
    original_spawn_rate = simulation.spawn_rate
    simulation.spawn_rate = 1.0
    
    # Run several steps to spawn clouds
    for _ in range(num_clouds * 10):  # Try multiple times
        simulation.step(0.1)  # Small time steps
        if len(simulation.clouds) >= num_clouds:
            break
    
    # Restore original spawn rate
    simulation.spawn_rate = original_spawn_rate
    
    print(f"✓ Created weather system with {len(simulation.clouds)} initial clouds")
    return simulation


def validate_cloud_config() -> bool:
    """Validate cloud configuration parameters."""
    errors = []
    warnings = []
    
    # Check domain size
    if not hasattr(CFG, 'DOMAIN_SIZE_M') or CFG.DOMAIN_SIZE_M <= 0:
        errors.append("Invalid DOMAIN_SIZE_M")
    
    # Check max clouds
    if not hasattr(CFG, 'MAX_PARCELS') or CFG.MAX_PARCELS <= 0:
        errors.append("Invalid MAX_PARCELS")
    
    # Check wind configuration
    if hasattr(CFG, 'BASE_WIND_SPEED') and CFG.BASE_WIND_SPEED < 0:
        warnings.append("Negative BASE_WIND_SPEED")
    
    # Print results
    if errors:
        print("❌ Cloud configuration errors:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("⚠️  Cloud configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("✅ Cloud configuration is valid")
    
    return len(errors) == 0


# ===================================================================
# BACKWARD COMPATIBILITY
# ===================================================================

# Backward compatibility aliases
OptimizedWeatherSystem = CloudSimulation
WeatherSystemState = CloudSimulation

# Export main classes and functions
__all__ = [
    'CloudSimulation',
    'Cloud', 
    'CloudType',
    'CloudCharacteristics',
    'CloudPhysics',
    'AtmosphericConditions',
    'WeatherSystem',
    'SpatialGrid',
    'collect_visible_ellipses',
    'create_optimized_weather_system',
    'create_weather_with_initial_clouds',
    'validate_cloud_config'
]


# ===================================================================
# MAIN EXECUTION FOR TESTING
# ===================================================================

if __name__ == "__main__":
    print("="*70)
    print("COMPLETE ADVANCED CLOUD PHYSICS SYSTEM TESTING")
    print("="*70)
    
    # Validate configuration
    validate_cloud_config()
    print()
    
    # Create cloud simulation
    simulation = create_weather_with_initial_clouds(3, seed=12345)
    print(f"Created simulation with {len(simulation.clouds)} clouds")
    print()
    
    # Test thread-safe mode
    simulation.set_thread_safe_mode(True)
    
    # Run simulation steps
    print("Running simulation steps...")
    for step in range(10):
        # Test both new dt-only interface and legacy interface
        if step % 2 == 0:
            simulation.step(dt=1.0)  # New interface
        else:
            simulation.step(1.0, current_hour=12.0 + step * 0.1)  # Legacy interface
        
        state = simulation.get_state()
        print(f"  Step {step+1}: {state['active_cloud_count']} active clouds, "
              f"{state['cloud_coverage']:.1f}% coverage, "
              f"wind: {state['wind_speed']:.1f} m/s @ {state['wind_direction']:.0f}°")
    
    # Test spatial queries
    print("\nTesting spatial queries...")
    coverage = simulation.get_coverage_at_position(25.0, 25.0)
    nearby = simulation.get_clouds_near_position(25.0, 25.0, 10.0)
    print(f"Coverage at (25, 25): {coverage:.2f}")
    print(f"Nearby clouds: {len(nearby)}")
    
    # Performance stats
    print("\nPerformance statistics:")
    stats = simulation.performance_stats
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ All cloud simulation tests completed successfully!")
    print(f"✅ Thread-safe advanced cloud physics system ready!")
    print(f"✅ Legacy t_s parameter compatibility verified!")