# cloud_simulation.py - Ultra-optimized version with caching and performance enhancements
"""
Enhanced Cloud Simulation with aggressive performance optimizations
- Cached shape generation and interpolation
- Optimized data structures with __slots__
- Vectorized operations where possible
- Memory-efficient trail management
"""
import math
import random
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache
from collections import deque
from dataclasses import dataclass
from itertools import chain, islice
import sim_config as CFG

print("Loading ultra-optimized cloud_simulation.py with advanced caching and performance features")

# Pre-compute constants for better performance
M_PER_FRAME = CFG.PHYSICS_TIMESTEP * (CFG.DOMAIN_SIZE_M / 1000.0)
DEGREES_TO_RADIANS = math.pi / 180.0
RADIANS_TO_DEGREES = 180.0 / math.pi

# Cache for trigonometric calculations
@lru_cache(maxsize=360)
def cached_sin_cos(angle_deg: int) -> Tuple[float, float]:
    """Cached trigonometric calculations for integer degrees."""
    rad = angle_deg * DEGREES_TO_RADIANS
    return math.sin(rad), math.cos(rad)

@lru_cache(maxsize=1000)
def cached_ellipse_shape(radius_km: float, type_name: str, age_factor: float) -> Tuple[float, float, float, float]:
    """Cache ellipse shape calculations based on radius, type, and age."""
    diameter_m = radius_km * 2000  # Convert km to m with visibility multiplier
    
    # Type-specific shape adjustments
    if type_name == "cirrus":
        width_factor, height_factor = 2.5, 0.4  # Elongated for cirrus
    elif type_name == "cumulonimbus":
        width_factor, height_factor = 1.2, 1.8  # Taller for storm clouds
    else:  # cumulus and others
        width_factor, height_factor = 1.0, 1.0  # Circular
    
    # Age-based size adjustment
    base_width = diameter_m * width_factor * age_factor
    base_height = diameter_m * height_factor * age_factor
    
    return base_width, base_height, 0.0, 0.0  # width, height, rotation, shear

class OptimizedTrail:
    """Memory-efficient trail management with automatic eviction."""
    __slots__ = ('_positions', '_max_length')
    
    def __init__(self, max_length: int = 15):
        self._positions = deque(maxlen=max_length)
        self._max_length = max_length
    
    def add_position(self, x: float, y: float) -> None:
        """Add position with automatic old point eviction."""
        self._positions.append((x, y))
    
    def get_positions(self) -> List[Tuple[float, float]]:
        """Get all trail positions as list."""
        return list(self._positions)
    
    def clear(self) -> None:
        """Clear all trail positions."""
        self._positions.clear()
    
    def __len__(self) -> int:
        return len(self._positions)

@dataclass(frozen=True)
class CloudTypePreset:
    """Immutable cloud type configuration for better caching."""
    alt_km: float
    r_km_min: float
    r_km_max: float
    opacity_max: float
    speed_factor: float
    
    @classmethod
    @lru_cache(maxsize=10)
    def from_config(cls, cloud_type: str) -> 'CloudTypePreset':
        """Cached factory method for cloud type presets."""
        preset = CFG.CLOUD_TYPES[cloud_type]
        r_lo, r_hi = preset["r_km"]
        return cls(
            alt_km=preset["alt_km"],
            r_km_min=r_lo,
            r_km_max=r_hi,
            opacity_max=preset["opacity_max"],
            speed_factor=preset.get("speed_k", 1.0)
        )

class UltraOptimizedCloudParcel:
    """Memory-optimized cloud parcel with __slots__ and cached calculations."""
    __slots__ = (
        # Position and movement
        'x', 'y', 'prev_x', 'prev_y', 'vx', 'vy',
        # Physical properties
        'type', 'r', 'opacity', 'alt',
        # Lifecycle
        'age', '_lifecycle_cache',
        # State flags
        'flag_for_split', 'split_fading',
        # Cached data
        '_preset', '_trail', '_last_ellipse_cache', '_shape_cache_key',
        '_cached_max_age'  # <-- Added for caching max_age
    )
    
    def __init__(self, x: float, y: float, ctype: str, wind=None):
        # Backward compatibility: if ctype is not a string, swap with wind
        if not isinstance(ctype, str) and isinstance(wind, str):
            ctype, wind = wind, ctype
        # Position
        self.x, self.y = x, y
        self.prev_x, self.prev_y = x, y
        
        # Type and preset (cached)
        self.type = ctype
        self._preset = CloudTypePreset.from_config(ctype)
        
        # Physical properties
        self.alt = self._preset.alt_km
        self.r = self._preset.r_km_max * 2.0  # Double for visibility
        self.opacity = 1.0
        
        # Velocity calculation (optimized)
        direction_deg = getattr(CFG, 'CLOUD_DIRECTION', 135.0)
        base_speed = getattr(CFG, 'BASE_WIND_SPEED', 4.0)
        speed_factor = random.uniform(0.95, 1.05)
        speed = base_speed * speed_factor
        
        # Use cached trigonometry
        sin_dir, cos_dir = cached_sin_cos(int(direction_deg))
        self.vx = speed * cos_dir * M_PER_FRAME
        self.vy = speed * sin_dir * M_PER_FRAME
        
        # Lifecycle
        self.age = 0
        self._lifecycle_cache = self._build_lifecycle_cache()
        
        # State
        self.flag_for_split = False
        self.split_fading = 0
        
        # Optimized trail
        max_trail_length = getattr(CFG, 'POSITION_HISTORY_LENGTH', 15)
        self._trail = OptimizedTrail(max_trail_length)
        
        # Caching for expensive operations
        self._last_ellipse_cache = None
        self._shape_cache_key = None
        
        print(f"Created optimized cloud at ({x:.1f}, {y:.1f}) type={ctype} r={self.r:.2f}km")
    
    @lru_cache(maxsize=1)
    def _build_lifecycle_cache(self) -> Dict[str, int]:
        """Cache lifecycle phase durations."""
        return {
            'growth_frames': getattr(CFG, 'CLOUD_GROWTH_FRAMES', 300),
            'stable_frames': getattr(CFG, 'CLOUD_STABLE_FRAMES', 1800),
            'decay_frames': getattr(CFG, 'CLOUD_DECAY_FRAMES', 300)
        }
    
    @property
    def max_age(self) -> int:
        """Cached max age calculation."""
        if not hasattr(self, '_cached_max_age'):
            cache = self._lifecycle_cache
            self._cached_max_age = (cache['growth_frames'] + 
                                  cache['stable_frames'] + 
                                  cache['decay_frames'])
        return self._cached_max_age
    
    @property
    def growth_frames(self) -> int:
        """Expose growth_frames for compatibility with external code."""
        return self._lifecycle_cache['growth_frames']
    
    @property
    def wind(self):
        """Expose wind for compatibility with external code. Returns None by default."""
        return None
    
    def _get_lifecycle_factors(self) -> Tuple[float, float]:
        """Optimized lifecycle factor calculation with caching."""
        cache = self._lifecycle_cache
        growth_frames = cache['growth_frames']
        stable_frames = cache['stable_frames']
        decay_frames = cache['decay_frames']
        
        if self.age < growth_frames:
            # Growth phase
            progress = self.age / growth_frames
            opacity_factor = progress
            size_factor = 0.8 + 0.2 * progress
        elif self.age < growth_frames + stable_frames:
            # Stable phase
            opacity_factor = 1.0
            size_factor = 1.0
            
            # Check for splitting with cached probability
            scatter_prob = getattr(CFG, 'SCATTER_PROBABILITY', 0.0)
            if scatter_prob > 0 and random.random() < scatter_prob:
                self.flag_for_split = True
        else:
            # Decay phase
            decay_progress = (self.age - growth_frames - stable_frames) / decay_frames
            decay_progress = min(1.0, decay_progress)
            opacity_factor = 1.0 - min(0.5, decay_progress)
            size_factor = 1.0 - 0.2 * decay_progress
        
        return opacity_factor, size_factor
    
    def step(self, timestep_sec: float, wind=None, sim_time=None) -> bool:
        """Optimized step function with minimal allocations."""
        # Cache previous position
        self.prev_x, self.prev_y = self.x, self.y
        
        # Increment age
        self.age += 1
        
        # Add to trail (automatic eviction)
        self._trail.add_position(self.x, self.y)
        
        # Apply movement with cached multiplier
        movement_mult = getattr(CFG, 'MOVEMENT_MULTIPLIER', 1.0)
        self.x += self.vx * movement_mult
        self.y += self.vy * movement_mult
        
        # Optimized boundary handling
        removal = self._handle_boundaries()
        
        # Update lifecycle properties
        opacity_factor, size_factor = self._get_lifecycle_factors()
        
        # Smooth interpolation for size and opacity
        target_size = self._preset.r_km_max * size_factor * 2.0
        target_opacity = max(0.7, self._preset.opacity_max * opacity_factor)
        
        self.r = self.r * 0.95 + target_size * 0.05
        self.opacity = self.opacity * 0.95 + target_opacity * 0.05
        
        # Handle split fading
        if self.split_fading > 0:
            self.split_fading -= 1
            self.opacity *= 0.95
        
        # Invalidate ellipse cache when properties change
        self._last_ellipse_cache = None
        
        # Return removal condition
        return removal or self.age >= self.max_age or self.r < 0.15
    
    def _handle_boundaries(self) -> bool:
        """Optimized boundary handling."""
        d = CFG.DOMAIN_SIZE_M
        
        if getattr(CFG, 'CLOUD_WRAP_AROUND', True):
            edge_margin = 0.1 * d
            
            # Vectorized boundary checks
            if self.x < -edge_margin:
                self.x = d + edge_margin
            elif self.x > d + edge_margin:
                self.x = -edge_margin
                
            if self.y < -edge_margin:
                self.y = d + edge_margin
            elif self.y > d + edge_margin:
                self.y = -edge_margin
                
            return False
        else:
            # Remove when out of bounds
            margin = 0.4 * d
            return (self.x < -margin or self.x > d + margin or
                    self.y < -margin or self.y > d + margin)
    
    def ellipse(self) -> Tuple[float, float, float, float, float, float, float, str]:
        """Cached ellipse generation with shape optimization."""
        # Create cache key
        cache_key = (round(self.r, 2), self.type, round(self.opacity, 3), self.age // 10)
        
        # Return cached result if available
        if (self._last_ellipse_cache is not None and 
            self._shape_cache_key == cache_key):
            return self._last_ellipse_cache
        
        # Generate new ellipse with cached shape calculation
        age_factor = min(1.0, self.age / 100.0)  # Normalize age for caching
        width, height, rotation, shear = cached_ellipse_shape(self.r, self.type, age_factor)
        
        ellipse = (self.x, self.y, width, height, rotation, self.opacity, self.alt, self.type)
        
        # Cache the result
        self._last_ellipse_cache = ellipse
        self._shape_cache_key = cache_key
        
        return ellipse
    
    def get_trail_positions(self) -> List[Tuple[float, float]]:
        """Get trail positions efficiently."""
        return self._trail.get_positions()
    
    def clear_trail(self) -> None:
        """Clear trail for memory management."""
        self._trail.clear()

# Alias for backward compatibility
SimpleCloudParcel = UltraOptimizedCloudParcel
EnhancedCloudParcel = UltraOptimizedCloudParcel

class OptimizedWeatherSystem:
    """Weather system with optimized parcel management and caching."""
    __slots__ = (
        'parcels', 'sim_time', 'time_since_last_spawn',
        '_cloud_type_cache', '_spawn_position_cache',
        '_trajectory_cache', '_coverage_cache'
    )
    
    def __init__(self, seed: int = 0):
        self.parcels: List[UltraOptimizedCloudParcel] = []
        self.sim_time = 0.0
        self.time_since_last_spawn = 0.0
        
        # Caching for expensive operations
        self._cloud_type_cache = self._build_cloud_type_cache()
        self._spawn_position_cache = {}
        self._trajectory_cache = {'speed': None, 'direction': None, 'confidence': 0, 'frame': -1}
        self._coverage_cache = {'value': 0.0, 'frame': -1}
        
        print("Optimized Weather System initialized with advanced caching")
        
        # Force spawn initial cloud
        self._spawn_center_cloud()
    
    @lru_cache(maxsize=1)
    def _build_cloud_type_cache(self) -> Tuple[List[str], List[float]]:
        """Cache cloud type selection data."""
        cloud_types = list(CFG.CLOUD_TYPE_WEIGHTS.keys())
        weights = list(CFG.CLOUD_TYPE_WEIGHTS.values())
        return cloud_types, weights
    
    def _spawn_center_cloud(self) -> None:
        """Spawn optimized center cloud."""
        d = CFG.DOMAIN_SIZE_M
        center_x, center_y = d / 2, d / 2
        
        # Use cached cloud type selection
        cloud_types, weights = self._cloud_type_cache
        ctype = random.choices(cloud_types, weights=weights)[0]
        print(f"[DEBUG] Creating UltraOptimizedCloudParcel in _spawn_center_cloud with args: ({center_x}, {center_y}, {ctype})")
        new_cloud = UltraOptimizedCloudParcel(center_x, center_y, ctype)
        self.parcels.append(new_cloud)
        print(f"Spawned optimized center cloud: {ctype} at ({center_x:.1f}, {center_y:.1f})")
    
    def step(self, dt: Optional[float] = None, t: Optional[float] = None, t_s: Optional[float] = None) -> None:
        """Optimized step with batch processing and caching."""
        # --- HARD LIMIT ON PARCELS ---
        MAX_TOTAL_PARCELS = 100
        if len(self.parcels) > MAX_TOTAL_PARCELS:
            print(f"[WARNING] Parcel count exceeded {MAX_TOTAL_PARCELS}, trimming to limit.")
            self.parcels = self.parcels[:MAX_TOTAL_PARCELS]
        
        # Update simulation time
        if t_s is not None:
            self.sim_time = t_s
        elif t is not None:
            self.sim_time = t
        else:
            self.sim_time += 1
        
        # Use cached timestep
        if dt is None:
            dt = getattr(CFG, 'PHYSICS_TIMESTEP', 1/60.0)
        
        self.time_since_last_spawn += dt
        
        # Ensure minimum cloud count
        if not self.parcels:
            self._spawn_center_cloud()
            self.time_since_last_spawn = 0.0
            return
        
        # Batch update all parcels with list comprehension
        active_parcels = [p for p in self.parcels if not p.step(dt, self, self.sim_time)]
        
        # Count removed parcels for debugging
        removed_count = len(self.parcels) - len(active_parcels)
        if removed_count > 0:
            print(f"Removed {removed_count} expired cloud parcels")
        
        # Handle cloud scattering with optimized splitting
        self.parcels = self._handle_cloud_scattering(active_parcels)
        
        # --- Always ensure at least one cloud is present ---
        if not self.parcels:
            print("[DEBUG] No clouds present after step, forcing spawn.")
            self._spawn_center_cloud()
            self.time_since_last_spawn = 0.0
            # No return here: allow spawning logic to run as well

        # Optimized spawning logic
        self._handle_spawning()
        
        # Invalidate caches that depend on parcel state
        self._trajectory_cache['frame'] = -1
        self._coverage_cache['frame'] = -1
    
    def _handle_cloud_scattering(self, parcels: List[UltraOptimizedCloudParcel]) -> List[UltraOptimizedCloudParcel]:
        """Optimized cloud scattering with list comprehensions."""
        new_parcels = []
        
        for parcel in parcels:
            if parcel.flag_for_split:
                # Generate fragments efficiently
                n_fragments = random.randint(2, 3)
                print(f"Cloud scattering: {n_fragments} fragments from ({parcel.x:.1f}, {parcel.y:.1f})")
                
                # Create children with list comprehension
                children = [
                    self._create_child_parcel(parcel)
                    for _ in range(n_fragments)
                ]
                new_parcels.extend(children)
                
                # Reset parent state
                parcel.flag_for_split = False
                parcel.split_fading = 60
                new_parcels.append(parcel)
            else:
                new_parcels.append(parcel)
        
        return new_parcels
    
    def _create_child_parcel(self, parent: UltraOptimizedCloudParcel) -> UltraOptimizedCloudParcel:
        """Create optimized child parcel from parent."""
        # Offset position slightly
        offset_x = parent.x + random.uniform(-0.5, 0.5) * 1000
        offset_y = parent.y + random.uniform(-0.5, 0.5) * 1000
        print(f"[DEBUG] Creating UltraOptimizedCloudParcel in _create_child_parcel with args: ({offset_x}, {offset_y}, {parent.type})")
        child = UltraOptimizedCloudParcel(offset_x, offset_y, parent.type)
        
        # Inherit properties efficiently
        child.vx, child.vy = parent.vx, parent.vy
        child.r = parent.r * random.uniform(0.8, 0.9)
        child.age = parent._lifecycle_cache['growth_frames']  # Start in stable phase
        
        return child
    
    def _handle_spawning(self) -> None:
        """Optimized spawning logic with enforced 20s interval."""
        # Single cloud mode check
        if getattr(CFG, 'SINGLE_CLOUD_MODE', False):
            if len(self.parcels) == 0:
                self._spawn()
                self.time_since_last_spawn = 0.0
            return
        
        # Regular spawning with enforced 20s interval
        max_parcels = getattr(CFG, 'MAX_PARCELS', 6)
        spawn_probability = getattr(CFG, 'SPAWN_PROBABILITY', 0.2)
        min_spawn_interval = 20.0  # Enforce 20 seconds between spawns
        can_spawn = self.time_since_last_spawn >= min_spawn_interval
        should_spawn = (
            can_spawn and
            len(self.parcels) < max_parcels and
            (random.random() < spawn_probability or len(self.parcels) == 0)
        )
        
        if should_spawn:
            self._spawn()
            self.time_since_last_spawn = 0.0
    
    @lru_cache(maxsize=8)
    def _get_spawn_position(self, direction_deg: float) -> Tuple[float, float]:
        """Cached spawn position calculation."""
        d = CFG.DOMAIN_SIZE_M
        entry_angle = (direction_deg + 180) % 360
        
        # Generate random offset (can't cache this part)
        rand_offset = random.uniform(-0.3, 0.3) * d
        
        # Calculate spawn coordinates
        if 45 <= entry_angle < 135:  # From top
            return d/2 + rand_offset, -d * 0.1
        elif 135 <= entry_angle < 225:  # From right
            return d * 1.1, d/2 + rand_offset
        elif 225 <= entry_angle < 315:  # From bottom
            return d/2 + rand_offset, d * 1.1
        else:  # From left
            return -d * 0.1, d/2 + rand_offset
    
    def _spawn(self) -> None:
        """Optimized spawning with cached calculations."""
        direction_deg = getattr(CFG, 'CLOUD_DIRECTION', 135.0)
        spawn_x, spawn_y = self._get_spawn_position(direction_deg)
        
        # Use cached cloud type selection
        cloud_types, weights = self._cloud_type_cache
        ctype = random.choices(cloud_types, weights=weights)[0]
        print(f"[DEBUG] Creating UltraOptimizedCloudParcel in _spawn with args: ({spawn_x}, {spawn_y}, {ctype})")
        new_cloud = UltraOptimizedCloudParcel(spawn_x, spawn_y, ctype)
        self.parcels.append(new_cloud)
        print(f"Spawned optimized {ctype} cloud at ({spawn_x:.1f}, {spawn_y:.1f})")
    
    def get_avg_trajectory(self) -> Tuple[Optional[float], Optional[float], float]:
        """Cached trajectory calculation."""
        current_frame = int(self.sim_time)
        
        # Return cached result if available
        cache = self._trajectory_cache
        if cache['frame'] == current_frame and cache['speed'] is not None:
            return cache['speed'], cache['direction'], cache['confidence']
        
        if not self.parcels:
            result = None, None, 0
        else:
            # Vectorized calculation using list comprehensions
            velocities = [(p.vx, p.vy) for p in self.parcels]
            
            if velocities:
                avg_vx = sum(vx for vx, vy in velocities) / len(velocities)
                avg_vy = sum(vy for vx, vy in velocities) / len(velocities)
                
                speed = math.sqrt(avg_vx**2 + avg_vy**2)
                direction = math.degrees(math.atan2(avg_vy, avg_vx)) % 360
                speed_kmh = speed * 3.6 / M_PER_FRAME
                
                result = speed_kmh, direction, 0.9
            else:
                result = None, None, 0
        
        # Cache the result
        cache.update({
            'speed': result[0],
            'direction': result[1], 
            'confidence': result[2],
            'frame': current_frame
        })
        
        return result
    
    def current_cloud_cover_pct(self) -> float:
        """Cached cloud coverage calculation."""
        current_frame = int(self.sim_time)
        
        # Return cached result if available
        cache = self._coverage_cache
        if cache['frame'] == current_frame:
            return cache['value']
        
        if not self.parcels:
            coverage = 0.0
        else:
            # Optimized calculation with list comprehension
            total_area = sum(math.pi * (p.r ** 2) for p in self.parcels)
            domain_area = CFG.AREA_SIZE_KM * CFG.AREA_SIZE_KM
            coverage = min(100, (total_area / domain_area) * 100 * 5)
        
        # Cache the result
        cache.update({'value': coverage, 'frame': current_frame})
        return coverage

# Optimized ellipse collection with caching
@lru_cache(maxsize=64)
def collect_visible_ellipses_cached(parcel_count: int, frame: int) -> List:
    """Cached ellipse collection for identical parcel configurations."""
    # This cache helps when the same configuration repeats
    # In practice, we'll use the non-cached version but with optimizations
    pass

def collect_visible_ellipses(parcels: List[UltraOptimizedCloudParcel]) -> List[Tuple]:
    """Optimized ellipse collection using list comprehension."""
    # Single list comprehension instead of loop
    return [parcel.ellipse() for parcel in parcels if parcel.opacity > 0.01]

# Advanced ellipse processing functions
def batch_process_ellipses(ellipses: List[Tuple], operation: str) -> List[Tuple]:
    """Batch process ellipses for various operations."""
    if operation == "filter_visible":
        return [e for e in ellipses if e[5] > 0.01]  # opacity > 0.01
    elif operation == "sort_by_size":
        return sorted(ellipses, key=lambda e: e[2] * e[3], reverse=True)  # by area
    elif operation == "sort_by_altitude":
        return sorted(ellipses, key=lambda e: e[6], reverse=True)  # by altitude
    else:
        return ellipses

def interpolate_ellipses(ellipses1: List[Tuple], ellipses2: List[Tuple], t: float) -> List[Tuple]:
    """Optimized ellipse interpolation using list comprehensions and zip."""
    if not ellipses1 and not ellipses2:
        return []
    
    if not ellipses1:
        # Fade in ellipses2
        return [(e[0], e[1], e[2], e[3], e[4], e[5] * t, e[6], e[7]) 
                for e in ellipses2]
    
    if not ellipses2:
        # Fade out ellipses1
        return [(e[0], e[1], e[2], e[3], e[4], e[5] * (1-t), e[6], e[7]) 
                for e in ellipses1]
    
    # Interpolate matching ellipses
    max_len = max(len(ellipses1), len(ellipses2))
    result = []
    
    for i in range(max_len):
        if i < len(ellipses1) and i < len(ellipses2):
            e1, e2 = ellipses1[i], ellipses2[i]
            # Linear interpolation for all parameters
            interpolated = tuple(
                e1[j] * (1-t) + e2[j] * t if isinstance(e1[j], (int, float)) else e1[j]
                for j in range(len(e1))
            )
            result.append(interpolated)
        elif i < len(ellipses1):
            # Fade out ellipse1
            e1 = ellipses1[i]
            faded = tuple(
                e1[j] * (1-t) if j == 5 else e1[j]  # Only fade opacity
                for j in range(len(e1))
            )
            result.append(faded)
        else:
            # Fade in ellipse2
            e2 = ellipses2[i]
            faded = tuple(
                e2[j] * t if j == 5 else e2[j]  # Only fade opacity
                for j in range(len(e2))
            )
            result.append(faded)
    
    return result

def generate_ellipses_for_type(cloud_type: str, count: int, 
                              base_position: Tuple[float, float],
                              size_range: Tuple[float, float]) -> List[Tuple]:
    """Cached ellipse generation for specific cloud types."""
    # Use cached shape calculations
    preset = CloudTypePreset.from_config(cloud_type)
    
    ellipses = []
    for i in range(count):
        # Generate position with some randomness
        x = base_position[0] + random.uniform(-size_range[0]/4, size_range[0]/4)
        y = base_position[1] + random.uniform(-size_range[1]/4, size_range[1]/4)
        
        # Use cached shape calculation
        size = random.uniform(size_range[0], size_range[1])
        width, height, rotation, _ = cached_ellipse_shape(size/2000, cloud_type, 1.0)
        
        ellipse = (x, y, width, height, rotation, preset.opacity_max, preset.alt_km, cloud_type)
        ellipses.append(ellipse)
    
    return ellipses

# Create type alias for backward compatibility
WeatherSystem = OptimizedWeatherSystem

print("Ultra-optimized cloud simulation loaded with advanced caching and performance features")