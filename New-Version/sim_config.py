# sim_config.py - Optimized production version
"""
Solar Farm Cloud Simulation Configuration

This module provides centralized configuration for the solar farm simulation.
All parameters are organized by system and automatically validated.
"""

import math
from typing import Dict, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from functools import lru_cache
import random

# ========== VERSION AND METADATA ==========
CONFIG_VERSION = "2.0.0"
CONFIG_DESCRIPTION = "Solar Farm Cloud Simulation Configuration"

# ========== CORE SIMULATION PARAMETERS ==========
class SimulationTiming:
    """Core timing parameters for the simulation."""
    FPS = 60
    PHYSICS_FPS = 30
    PHYSICS_TIMESTEP = 1.0 / PHYSICS_FPS
    LOGIC_FPS = 60
    QUALITY_FPS_THRESHOLD = 0.9

class DomainSettings:
    """Physical domain and area settings."""
    DOMAIN_SIZE_M = 50_000
    AREA_SIZE_KM = 50.0
    SPAWN_BUFFER_M = 1_000
    PANEL_SIZE_KM = 0.4
    DOMAIN_MARGIN_PCT = 0.4

# ========== WIND SYSTEM CONFIGURATION ==========
@dataclass(frozen=True)
class WindConfig:
    """Immutable wind configuration with computed properties."""
    base_speed_ms: float = 6.0
    base_direction_deg: float = 135.0
    movement_multiplier: float = 1.5
    update_interval_sec: float = 0.25
    smooth_factor: float = 0.98
    turbulence: float = 0.05
    use_wind_field: bool = False
    
    # Wind effects
    gust_probability: float = 0.01
    gust_duration_frames: int = 180
    diurnal_enabled: bool = True
    
    @property
    @lru_cache(maxsize=1)
    def speed_kmh(self) -> float:
        """Wind speed in km/h."""
        return self.base_speed_ms * 3.6
    
    @property
    @lru_cache(maxsize=1)
    def direction_rad(self) -> float:
        """Wind direction in radians."""
        return math.radians(self.base_direction_deg)
    
    @property
    @lru_cache(maxsize=1)
    def velocity_vector_ms(self) -> Tuple[float, float]:
        """Wind velocity vector in m/s (vx, vy)."""
        return (
            self.base_speed_ms * math.cos(self.direction_rad),
            self.base_speed_ms * math.sin(self.direction_rad)
        )
    
    @property
    @lru_cache(maxsize=1)
    def velocity_vector_kmh(self) -> Tuple[float, float]:
        """Wind velocity vector in km/h (vx, vy)."""
        vx_ms, vy_ms = self.velocity_vector_ms
        return (vx_ms * 3.6, vy_ms * 3.6)

# Global wind configuration instance
WIND = WindConfig()

# ========== WIND LAYER SYSTEM ==========
class WindLayers:
    """Wind layer configuration with automatic generation."""
    HEIGHTS = [0.0, 0.25, 1.0, 8.0, 15.0]  # Added 15.0 to fix config validation
    SPEED_FACTORS = [1.2, 1.0, 0.8, 0.6]
    GRID_SIZE = 256
    INTEGRATION_DEGREE = 4
    
    @classmethod
    @lru_cache(maxsize=1)
    def direction_offsets(cls) -> list:
        """Generate consistent direction offsets for all layers."""
        return [0.0] * len(cls.SPEED_FACTORS)
    
    @classmethod
    def validate_layers(cls) -> bool:
        """Validate layer configuration consistency."""
        return len(cls.HEIGHTS) == len(cls.SPEED_FACTORS) + 1

# ========== CLOUD SYSTEM CONFIGURATION ==========
@dataclass(frozen=True)
class CloudPopulationConfig:
    """Cloud population and spawning control."""
    max_parcels: int = 6
    spawn_probability: float = 0.50
    single_cloud_mode: bool = False
    force_initial_cloud: bool = True
    min_spawn_interval: float = 10.0
    wrap_around: bool = True
    scatter_probability: float = 0.20
    scatter_fragments: Tuple[int, int] = (2, 4)

@dataclass(frozen=True)
class CloudLifecycleConfig:
    """Cloud lifecycle timing configuration."""
    growth_frames: int = 60      # 1 second at 60 FPS
    stable_frames: int = 1800    # 30 seconds at 60 FPS
    decay_frames: int = 300      # 5 seconds at 60 FPS
    breathing_amplitude: float = 0.05
    random_removal_chance: float = 0.00001
    
    @property
    def total_lifetime_frames(self) -> int:
        """Total cloud lifetime in frames."""
        return self.growth_frames + self.stable_frames + self.decay_frames
    
    @property
    def total_lifetime_seconds(self) -> float:
        """Total cloud lifetime in seconds."""
        return self.total_lifetime_frames / SimulationTiming.FPS

@dataclass(frozen=True)
class CloudAppearanceConfig:
    """Cloud visual appearance configuration."""
    opacity_ramp: float = 0.01
    max_opacity: float = 0.95
    min_altitude: float = 0.2
    max_altitude: float = 5.0
    parallax_factor: float = 0.2
    max_radius_km: float = 6.0
    position_history_length: int = 15

# Global cloud configuration instances
CLOUD_POPULATION = CloudPopulationConfig()
CLOUD_LIFECYCLE = CloudLifecycleConfig()
CLOUD_APPEARANCE = CloudAppearanceConfig()

# ========== CLOUD TYPE SYSTEM ==========
class CloudTypeDefinition(NamedTuple):
    """Immutable cloud type definition."""
    altitude_km: float
    radius_range_km: Tuple[float, float]
    opacity_max: float
    speed_factor: float
    size_range: Tuple[int, int]  # (width_min, width_max) in meters
    height_range: Tuple[int, int]  # (height_min, height_max) in meters
    opacity_range: Tuple[float, float]
    rotation_range: Tuple[float, float]

class CloudTypeRegistry:
    """Registry for cloud type definitions with computed properties."""
    
    _DEFINITIONS = {
        "cirrus": CloudTypeDefinition(
            altitude_km=8.0,
            radius_range_km=(2.0, 3.5),
            opacity_max=0.80,
            speed_factor=0.5,
            size_range=(2500, 5000),
            height_range=(500, 1000),
            opacity_range=(0.5, 0.8),
            rotation_range=(-0.3, 0.3)
        ),
        "cumulus": CloudTypeDefinition(
            altitude_km=1.5,
            radius_range_km=(1.5, 3.0),
            opacity_max=0.95,
            speed_factor=1.0,
            size_range=(2000, 3500),
            height_range=(2000, 3500),
            opacity_range=(0.7, 0.95),
            rotation_range=(-0.1, 0.1)
        ),
        "cumulonimbus": CloudTypeDefinition(
            altitude_km=2.0,
            radius_range_km=(4.0, 6.0),
            opacity_max=0.98,
            speed_factor=0.8,
            size_range=(3000, 4000),
            height_range=(4000, 6000),
            opacity_range=(0.85, 1.0),
            rotation_range=(-0.05, 0.05)
        )
    }
    
    # Selection weights for random cloud type generation
    WEIGHTS = {"cirrus": 2, "cumulus": 6, "cumulonimbus": 1}
    
    @classmethod
    def get_definition(cls, cloud_type: str) -> CloudTypeDefinition:
        """Get cloud type definition with fallback to cumulus."""
        return cls._DEFINITIONS.get(cloud_type, cls._DEFINITIONS["cumulus"])
    
    @classmethod
    def get_all_types(cls) -> list:
        """Get list of all cloud types."""
        return list(cls._DEFINITIONS.keys())
    
    @classmethod
    @lru_cache(maxsize=None)
    def get_puff_range(cls, cloud_type: str) -> Tuple[int, int]:
        """Compute puff count range based on cloud characteristics."""
        definition = cls.get_definition(cloud_type)
        avg_radius = sum(definition.radius_range_km) / 2
        
        puff_multipliers = {
            "cirrus": (0.8, 1.0),
            "cumulus": (1.5, 2.0),
            "cumulonimbus": (2.0, 2.5)
        }
        
        min_mult, max_mult = puff_multipliers.get(cloud_type, (1.0, 1.5))
        base_counts = {"cirrus": 3, "cumulus": 5, "cumulonimbus": 10}
        
        min_puffs = max(base_counts.get(cloud_type, 5), int(avg_radius * min_mult))
        max_puffs = max(min_puffs + 1, int(avg_radius * max_mult))
        
        return (min_puffs, max_puffs)

# ========== SHADOW AND POWER SYSTEM ==========
@dataclass(frozen=True)
class ShadowConfig:
    """Shadow calculation and power system configuration."""
    transmittance: float = 0.2
    fade_duration_ms: int = 500
    penumbra_width_m: int = 60

SHADOWS = ShadowConfig()

# ========== BACKWARD COMPATIBILITY LAYER ==========
class CompatibilityLayer:
    """Provides backward compatibility with old configuration format."""
    
    @staticmethod
    @lru_cache(maxsize=None)
    def _generate_legacy_constants():
        """Generate legacy constant mappings."""
        # Basic simulation constants
        constants = {
            'FPS': SimulationTiming.FPS,
            'PHYSICS_FPS': SimulationTiming.PHYSICS_FPS,
            'PHYSICS_TIMESTEP': SimulationTiming.PHYSICS_TIMESTEP,
            'DOMAIN_SIZE_M': DomainSettings.DOMAIN_SIZE_M,
            'AREA_SIZE_KM': DomainSettings.AREA_SIZE_KM,
            'PANEL_SIZE_KM': DomainSettings.PANEL_SIZE_KM,
            
            # Wind constants
            'BASE_WIND_SPEED': WIND.base_speed_ms,
            'BASE_WIND_DIRECTION': WIND.base_direction_deg,
            'DEFAULT_WIND_SPEED': WIND.base_speed_ms,
            'DEFAULT_WIND_DIRECTION': WIND.base_direction_deg,
            'DEFAULT_WIND_SPEED_KMH': WIND.speed_kmh,
            'DEFAULT_WIND_DIRECTION_DEG': WIND.base_direction_deg,
            'MOVEMENT_MULTIPLIER': WIND.movement_multiplier,
            'WIND_UPDATE_SEC': WIND.update_interval_sec,
            'WIND_TURBULENCE': WIND.turbulence,
            'USE_WIND_FIELD': WIND.use_wind_field,
            
            # Wind layers
            'LAYER_HEIGHTS': WindLayers.HEIGHTS,
            'LAYER_SPEED_FACTORS': WindLayers.SPEED_FACTORS,
            'LAYER_DIRECTION_OFFSETS': WindLayers.direction_offsets(),
            'WIND_GRID': WindLayers.GRID_SIZE,
            
            # Cloud constants
            'MAX_PARCELS': CLOUD_POPULATION.max_parcels,
            'SPAWN_PROBABILITY': CLOUD_POPULATION.spawn_probability,
            'CLOUD_GROWTH_FRAMES': CLOUD_LIFECYCLE.growth_frames,
            'CLOUD_STABLE_FRAMES': CLOUD_LIFECYCLE.stable_frames,
            'CLOUD_DECAY_FRAMES': CLOUD_LIFECYCLE.decay_frames,
            'SCATTER_PROBABILITY': CLOUD_POPULATION.scatter_probability,
            
            # Shadow constants
            'CLOUD_TRANSMITTANCE': SHADOWS.transmittance,
            'SHADOW_FADE_MS': SHADOWS.fade_duration_ms,
            'PENUMBRA_WIDTH': SHADOWS.penumbra_width_m,
        }
        
        # Cloud type dictionaries
        constants['CLOUD_TYPES'] = {
            cloud_type: {
                "alt_km": defn.altitude_km,
                "r_km": defn.radius_range_km,
                "opacity_max": defn.opacity_max,
                "speed_k": defn.speed_factor
            }
            for cloud_type, defn in CloudTypeRegistry._DEFINITIONS.items()
        }
        
        constants['CLOUD_TYPE_PROFILES'] = {
            cloud_type: {
                "cw": defn.size_range,
                "ch": defn.height_range,
                "opacity": defn.opacity_range,
                "rotation": defn.rotation_range
            }
            for cloud_type, defn in CloudTypeRegistry._DEFINITIONS.items()
        }
        
        constants['CLOUD_TYPE_WEIGHTS'] = CloudTypeRegistry.WEIGHTS
        
        # Puff ranges
        constants['PUFF_MIN'] = {
            cloud_type: CloudTypeRegistry.get_puff_range(cloud_type)[0]
            for cloud_type in CloudTypeRegistry.get_all_types()
        }
        constants['PUFF_MAX'] = {
            cloud_type: CloudTypeRegistry.get_puff_range(cloud_type)[1]
            for cloud_type in CloudTypeRegistry.get_all_types()
        }
        
        return constants
    
    def __getattr__(self, name):
        """Dynamic attribute access for backward compatibility."""
        constants = self._generate_legacy_constants()
        if name in constants:
            return constants[name]
        raise AttributeError(f"No configuration parameter named '{name}'")

# Create compatibility instance
_compat = CompatibilityLayer()

# Export legacy constants dynamically
def __getattr__(name):
    """Module-level attribute access for backward compatibility."""
    return getattr(_compat, name)

# ========== CONFIGURATION VALIDATION ==========
class ConfigValidator:
    """Validates configuration consistency and constraints."""
    
    @staticmethod
    def validate_all() -> Dict[str, list]:
        """Run all validation checks."""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Critical validations (errors)
        if SimulationTiming.PHYSICS_TIMESTEP <= 0:
            issues['errors'].append("PHYSICS_TIMESTEP must be positive")
        
        if DomainSettings.DOMAIN_SIZE_M <= 0:
            issues['errors'].append("DOMAIN_SIZE_M must be positive")
        
        if not WindLayers.validate_layers():
            issues['errors'].append("Wind layer configuration inconsistent")
        
        # Non-critical validations (warnings)
        if WIND.base_speed_ms > 20.0:
            issues['warnings'].append("Wind speed unusually high (>20 m/s)")
        
        if CLOUD_POPULATION.max_parcels > 20:
            issues['warnings'].append("High cloud count may impact performance")
        
        total_weights = sum(CloudTypeRegistry.WEIGHTS.values())
        if abs(total_weights - 9) > 0.1:  # Expected total: 2+6+1=9
            issues['warnings'].append("Cloud type weights may be unbalanced")
        
        return issues
    
    @staticmethod
    def print_validation_report():
        """Print human-readable validation report."""
        issues = ConfigValidator.validate_all()
        
        if issues['errors']:
            print("❌ Configuration Errors:")
            for error in issues['errors']:
                print(f"  • {error}")
        
        if issues['warnings']:
            print("⚠️  Configuration Warnings:")
            for warning in issues['warnings']:
                print(f"  • {warning}")
        
        if not issues['errors'] and not issues['warnings']:
            print("✅ Configuration validation passed")

# ========== UTILITY FUNCTIONS ==========
def get_effective_wind_speed(altitude_km: float = 1.0) -> float:
    """Get effective wind speed at given altitude."""
    # Find appropriate layer
    for i, height in enumerate(WindLayers.HEIGHTS[:-1]):
        if altitude_km >= height and altitude_km < WindLayers.HEIGHTS[i + 1]:
            factor = WindLayers.SPEED_FACTORS[i]
            return WIND.base_speed_ms * factor
    
    # Default to highest layer
    return WIND.base_speed_ms * WindLayers.SPEED_FACTORS[-1]

def get_cloud_config(cloud_type: str) -> CloudTypeDefinition:
    """Get complete cloud type configuration."""
    return CloudTypeRegistry.get_definition(cloud_type)

def print_config_summary():
    """Print a summary of key configuration parameters."""
    print(f"=== Solar Farm Simulation Config v{CONFIG_VERSION} ===")
    print(f"Domain: {DomainSettings.DOMAIN_SIZE_M}m ({DomainSettings.AREA_SIZE_KM}km)")
    print(f"Wind: {WIND.base_speed_ms}m/s @ {WIND.base_direction_deg}°")
    print(f"Clouds: Max {CLOUD_POPULATION.max_parcels}, Types: {len(CloudTypeRegistry._DEFINITIONS)}")
    print(f"Physics: {SimulationTiming.PHYSICS_FPS}fps timestep")

# ========== MODULE INITIALIZATION ==========
if __name__ == "__main__":
    # Run validation and print summary when module is executed directly
    print_config_summary()
    ConfigValidator.print_validation_report()
else:
    # Validate on import (optional - remove if too strict)
    issues = ConfigValidator.validate_all()
    print(issues)
    if issues['errors']:
        import warnings
        warnings.warn(f"Configuration has {len(issues['errors'])} errors", UserWarning)

# ===== QUICK CLOUD SPAWN FOR DEBUGGING =====
# Force clouds to spawn and move for testing
CLOUD_POPULATION = CloudPopulationConfig(
    max_parcels=6,
    spawn_probability=0.8,  # High probability for frequent spawning
    single_cloud_mode=False,
    force_initial_cloud=True,
    min_spawn_interval=1.0,  # Lower interval for more frequent spawns
    wrap_around=True,
    scatter_probability=0.20,
    scatter_fragments=(2, 4)
)

# ===== DEBUG: Print cloud spawn and parcel info =====
# Patch OptimizedWeatherSystem to print cloud spawn and parcel status
import simulation_controller as _sim_ctrl
import sim_config as CFG
print("[DEBUG] Patch for OptimizedWeatherSystem._spawn is active")
if hasattr(_sim_ctrl, 'OptimizedWeatherSystem'):
    orig_spawn = _sim_ctrl.OptimizedWeatherSystem._spawn
    def debug_spawn(self, t):
        print(f"[DEBUG] _spawn called with t={t}")
        try:
            print(f"[DEBUG] Parcels before spawn: {len(self.parcels)}")
            orig_spawn(self, t)
            print(f"[DEBUG] Parcels after spawn: {len(self.parcels)}")
            for i, p in enumerate(self.parcels):
                print(f"  Parcel {i}: pos=({getattr(p, 'x', None):.1f}, {getattr(p, 'y', None):.1f}), vx={getattr(p, 'vx', None)}, vy={getattr(p, 'vy', None)}, type={getattr(p, 'type', None)}")
        except Exception as e:
            print(f"[DEBUG] Exception in _spawn: {e}")
    _sim_ctrl.OptimizedWeatherSystem._spawn = debug_spawn
    # Also patch step to print should_spawn
    orig_step = _sim_ctrl.OptimizedWeatherSystem.step
    def debug_step(self, t=None, dt=None, t_s=None):
        # Copy the logic up to should_spawn
        if t_s is not None:
            self.sim_time = t_s
        elif t is not None:
            self.sim_time = t
        else:
            self.sim_time += 1
        if dt is None:
            dt = getattr(CFG, 'PHYSICS_TIMESTEP', 1/60.0)
        self.update_weather_pattern()
        if hasattr(self.wind, 'step'):
            self.wind.step(self.sim_time)
        if self.parcels:
            expired_indices = []
            for i, parcel in enumerate(self.parcels):
                if hasattr(parcel, 'step'):
                    if parcel.step(dt, self.wind, self.sim_time):
                        expired_indices.append(i)
            if expired_indices:
                for i in reversed(expired_indices):
                    self.parcels.pop(i)
        if self.parcels:
            new_parcels = []
            for p in self.parcels:
                if getattr(p, "flag_for_split", False):
                    n = random.randint(2, 3)
                    for _ in range(n):
                        try:
                            from cloud_simulation import EnhancedCloudParcel
                            child = p.__class__(
                                p.x + random.uniform(-500, 500),
                                p.y + random.uniform(-500, 500),
                                p.type)
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
        self.time_since_last_spawn += dt
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
        print(f"[DEBUG] step: parcels={len(self.parcels)}, should_spawn={should_spawn}")
        if should_spawn:
            self._spawn(self.sim_time)
            self.time_since_last_spawn = 0.0
        # Call the rest of the original step
        return orig_step(self, t, dt, t_s)
    _sim_ctrl.OptimizedWeatherSystem.step = debug_step