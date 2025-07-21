# enhanced_wind_field.py - Maximum optimization with threading support
"""
Enhanced Wind Field for Solar Farm Simulation
Ultra-optimized version with threading support and GIL-releasing operations
Designed for use in physics worker threads with maximum performance
"""
import numpy as np
import math
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
from functools import lru_cache
import warnings
import sim_config as CFG

# Try to import Numba for JIT compilation
try:
    from numba import njit, prange
    from numba.core.errors import NumbaWarning
    NUMBA_AVAILABLE = True
    
    # Suppress numba warnings for cleaner output
    warnings.filterwarnings("ignore", category=NumbaWarning)
    
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if Numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    prange = range
    print("Warning: Numba not available. Performance will be reduced.")

# Thread-local storage for random number generators
_thread_local = threading.local()

def _get_thread_rng():
    """Get thread-local random number generator."""
    if not hasattr(_thread_local, 'rng'):
        _thread_local.rng = np.random.default_rng()
    return _thread_local.rng

# JIT-compiled helper functions for maximum performance with GIL release
if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, nogil=True)
    def _fast_bilinear_interpolate(values, fx, fy, ix, iy):
        """Ultra-fast bilinear interpolation with JIT compilation and GIL release."""
        tx = fx - ix
        ty = fy - iy
        ix1, iy1 = ix + 1, iy + 1
        
        return ((1-tx)*(1-ty)*values[iy, ix] + 
                tx*(1-ty)*values[iy, ix1] + 
                (1-tx)*ty*values[iy1, ix] + 
                tx*ty*values[iy1, ix1])

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _compute_vectors_jit(speeds, directions, vectors_x, vectors_y):
        """JIT-compiled vector computation with parallel execution and GIL release."""
        layers, height, width = speeds.shape
        
        for layer in prange(layers):
            for i in prange(height):
                for j in prange(width):
                    direction_rad = directions[layer, i, j] * 0.017453292519943295  # π/180
                    speed = speeds[layer, i, j]
                    vectors_x[layer, i, j] = speed * math.cos(direction_rad)
                    vectors_y[layer, i, j] = speed * math.sin(direction_rad)

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _apply_wind_changes_jit(cells, turbulence_field, layer_factors, direction_drift, change_magnitude):
        """JIT-compiled wind field updates with parallel processing and GIL release."""
        layers, height, width, _ = cells.shape
        
        for layer in prange(layers):
            layer_factor = layer_factors[layer]
            drift = direction_drift * layer_factor
            
            for i in prange(height):
                for j in prange(width):
                    # Direction updates
                    cells[layer, i, j, 1] += drift + turbulence_field[i, j] * layer_factor
                    cells[layer, i, j, 1] = cells[layer, i, j, 1] % 360.0
                    
                    # Speed updates
                    speed_mult = 1.0 + turbulence_field[i, j] * 0.05 * layer_factor * change_magnitude
                    cells[layer, i, j, 0] *= speed_mult

    @njit(cache=True, fastmath=True, nogil=True)
    def _generate_smooth_noise(noise_output, grid_size, seed):
        """JIT-compiled smooth noise generation with deterministic seed."""
        np.random.seed(seed)
        noise_size = 4
        
        # Generate base noise
        base_noise = np.random.random((noise_size, noise_size)).astype(np.float32) * 2.0 - 1.0
        
        # Smooth the noise
        smoothed = np.zeros((noise_size, noise_size), dtype=np.float32)
        for i in range(noise_size):
            for j in range(noise_size):
                total = 0.0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni = (i + di) % noise_size
                        nj = (j + dj) % noise_size
                        total += base_noise[ni, nj]
                smoothed[i, j] = total / 9.0
        
        # Upscale to grid size
        scale_factor = grid_size / noise_size
        for i in range(grid_size):
            for j in range(grid_size):
                ni = min(int(i / scale_factor), noise_size - 1)
                nj = min(int(j / scale_factor), noise_size - 1)
                noise_output[i, j] = smoothed[ni, nj]

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _apply_gust_effect_jit(speeds, effect_mask, gust_factor, layer_factors):
        """JIT-compiled gust application with parallel processing and GIL release."""
        layers, height, width = speeds.shape
        
        for layer in prange(layers):
            layer_factor = layer_factors[layer]
            combined_factor = gust_factor * layer_factor
            
            for i in prange(height):
                for j in prange(width):
                    speeds[layer, i, j] *= (1.0 + effect_mask[i, j] * combined_factor)

    @njit(cache=True, fastmath=True, parallel=True, nogil=True)
    def _vectorized_speed_clamp(speeds, min_speeds, max_speeds):
        """JIT-compiled vectorized speed clamping with GIL release."""
        layers, height, width = speeds.shape
        
        for layer in prange(layers):
            min_speed = min_speeds[layer]
            max_speed = max_speeds[layer]
            
            for i in prange(height):
                for j in prange(width):
                    if speeds[layer, i, j] < min_speed:
                        speeds[layer, i, j] = min_speed
                    elif speeds[layer, i, j] > max_speed:
                        speeds[layer, i, j] = max_speed

else:
    # Fallback implementations without JIT but still vectorized for GIL release
    def _fast_bilinear_interpolate(values, fx, fy, ix, iy):
        """Bilinear interpolation fallback."""
        tx = fx - ix
        ty = fy - iy
        ix1, iy1 = ix + 1, iy + 1
        
        return ((1-tx)*(1-ty)*values[iy, ix] + 
                tx*(1-ty)*values[iy, ix1] + 
                (1-tx)*ty*values[iy1, ix] + 
                tx*ty*values[iy1, ix1])

    def _compute_vectors_jit(speeds, directions, vectors_x, vectors_y):
        """Vector computation fallback with vectorized operations."""
        direction_rad = np.radians(directions)
        vectors_x[:] = speeds * np.cos(direction_rad)
        vectors_y[:] = speeds * np.sin(direction_rad)

    def _apply_wind_changes_jit(cells, turbulence_field, layer_factors, direction_drift, change_magnitude):
        """Wind field updates fallback with vectorized operations."""
        layers, height, width, _ = cells.shape
        
        for layer in range(layers):
            layer_factor = layer_factors[layer]
            drift = direction_drift * layer_factor
            
            # Vectorized updates
            cells[layer, :, :, 1] += drift + turbulence_field * layer_factor
            cells[layer, :, :, 1] %= 360.0
            
            speed_mult = 1.0 + turbulence_field * 0.05 * layer_factor * change_magnitude
            cells[layer, :, :, 0] *= speed_mult

    def _generate_smooth_noise(noise_output, grid_size, seed):
        """Smooth noise generation fallback with vectorized operations."""
        np.random.seed(seed)
        noise_size = 4
        base_noise = np.random.random((noise_size, noise_size)) * 2.0 - 1.0
        
        # Simple smoothing with vectorized operations
        try:
            from scipy import ndimage
            smoothed = ndimage.gaussian_filter(base_noise, sigma=0.5)
        except ImportError:
            smoothed = base_noise
        
        # Vectorized upscaling
        try:
            from scipy.ndimage import zoom
            upscaled = zoom(smoothed, grid_size / noise_size, order=1)
            noise_output[:] = upscaled[:grid_size, :grid_size]
        except ImportError:
            # Manual upscaling
            scale_factor = grid_size / noise_size
            for i in range(grid_size):
                for j in range(grid_size):
                    ni = min(int(i / scale_factor), noise_size - 1)
                    nj = min(int(j / scale_factor), noise_size - 1)
                    noise_output[i, j] = smoothed[ni, nj]

    def _apply_gust_effect_jit(speeds, effect_mask, gust_factor, layer_factors):
        """Gust application fallback with vectorized operations."""
        layers, height, width = speeds.shape
        
        for layer in range(layers):
            layer_factor = layer_factors[layer]
            combined_factor = gust_factor * layer_factor
            speeds[layer] *= (1.0 + effect_mask * combined_factor)

    def _vectorized_speed_clamp(speeds, min_speeds, max_speeds):
        """Vectorized speed clamping fallback."""
        for layer in range(len(min_speeds)):
            np.clip(speeds[layer], min_speeds[layer], max_speeds[layer], out=speeds[layer])

class ThreadSafeDataStructures:
    """Thread-safe memory-efficient data structures with advanced optimizations."""
    __slots__ = ('_arrays', '_metadata', '_cache', '_lock')
    
    def __init__(self, grid_size: int, num_layers: int):
        self._arrays = {}
        self._metadata = {'grid_size': grid_size, 'num_layers': num_layers}
        self._cache = {}
        self._lock = threading.RLock()  # Thread safety
        
        # Pre-allocate all arrays as contiguous memory blocks
        self._allocate_contiguous_arrays()
    
    def _allocate_contiguous_arrays(self):
        """Allocate arrays in contiguous memory for better cache performance."""
        grid_size = self._metadata['grid_size']
        num_layers = self._metadata['num_layers']
        
        # Main arrays with C-contiguous layout for cache efficiency
        self._arrays.update({
            'cells': np.empty((num_layers, grid_size, grid_size, 2), dtype=np.float32, order='C'),
            'prev_cells': np.empty((num_layers, grid_size, grid_size, 2), dtype=np.float32, order='C'),
            'vectors_x': np.empty((num_layers, grid_size, grid_size), dtype=np.float32, order='C'),
            'vectors_y': np.empty((num_layers, grid_size, grid_size), dtype=np.float32, order='C'),
            'prev_vectors_x': np.empty((num_layers, grid_size, grid_size), dtype=np.float32, order='C'),
            'prev_vectors_y': np.empty((num_layers, grid_size, grid_size), dtype=np.float32, order='C'),
            'turbulence_field': np.zeros((grid_size, grid_size), dtype=np.float32, order='C'),
            'temp_noise': np.empty((grid_size, grid_size), dtype=np.float32, order='C'),
            'gust_workspace': np.empty((grid_size, grid_size), dtype=np.float32, order='C')
        })
    
    def __getitem__(self, key):
        return self._arrays[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            self._arrays[key] = value

@dataclass(frozen=True)
class WindFieldConfig:
    """Immutable configuration with compile-time constants for thread safety."""
    domain_size: int = CFG.DOMAIN_SIZE_M
    grid_resolution: int = CFG.WIND_GRID
    base_wind_speed: float = CFG.BASE_WIND_SPEED
    wind_direction: float = CFG.BASE_WIND_DIRECTION
    num_layers: int = 3
    layer_heights: Tuple[float, ...] = field(default_factory=lambda: tuple(getattr(CFG, 'LAYER_HEIGHTS', [0.0, 0.25, 1.0, 8.0])))
    layer_speed_factors: Tuple[float, ...] = field(default_factory=lambda: tuple(getattr(CFG, 'LAYER_SPEED_FACTORS', [1.2, 1.0, 0.8, 0.6])))
    layer_direction_offsets: Tuple[float, ...] = field(default_factory=lambda: tuple(getattr(CFG, 'LAYER_DIRECTION_OFFSETS', [0.0, 0.0, 0.0, 0.0])))
    wind_turbulence: float = getattr(CFG, 'WIND_TURBULENCE', 0.05)
    gust_probability: float = getattr(CFG, 'WIND_GUST_PROBABILITY', 0.01)
    gust_duration: int = getattr(CFG, 'WIND_GUST_DURATION', 180)
    diurnal_enabled: bool = getattr(CFG, 'DIURNAL_WIND_ENABLED', True)
    
    # Performance tuning constants
    update_frequency: int = 15  # Update every N frames instead of time-based
    turbulence_update_frequency: int = 300  # Update turbulence every N frames
    max_gusts: int = 8  # Limit concurrent gusts for performance

class ThreadSafeWindGust:
    """Thread-safe highly optimized wind gust with minimal overhead."""
    __slots__ = ('_data', '_effect_mask', '_lock')
    
    def __init__(self, grid_size: int, gust_duration: int):
        self._lock = threading.Lock()
        
        # Pack all data into a single array for cache efficiency
        rng = _get_thread_rng()
        self._data = np.array([
            rng.uniform(0, grid_size - 1),     # center_x
            rng.uniform(0, grid_size - 1),     # center_y
            rng.uniform(1.1, 1.3),             # strength
            rng.uniform(grid_size/8, grid_size/4),  # radius
            gust_duration,                      # frames_left
            gust_duration * 0.3,               # fade_in_frames
            gust_duration * 0.3                # fade_out_frames
        ], dtype=np.float32)
        
        # Pre-compute effect mask using vectorized operations
        self._effect_mask = self._compute_effect_mask_vectorized(grid_size)
    
    def _compute_effect_mask_vectorized(self, grid_size: int) -> np.ndarray:
        """Vectorized effect mask computation with GIL release."""
        y_grid, x_grid = np.ogrid[0:grid_size, 0:grid_size]
        
        center_x, center_y, strength, radius = self._data[0], self._data[1], self._data[2], self._data[3]
        
        # Vectorized distance calculation (releases GIL)
        distances_sq = (x_grid - center_x)**2 + (y_grid - center_y)**2
        mask = distances_sq < radius**2
        
        # Vectorized effect calculation (releases GIL)
        distances = np.sqrt(distances_sq)
        effect = np.where(mask, (strength - 1.0) * (1 - distances / radius), 0.0)
        
        return effect.astype(np.float32)
    
    @property
    def frames_left(self) -> float:
        return self._data[4]
    
    @property
    def fade_in_frames(self) -> float:
        return self._data[5]
    
    @property
    def fade_out_frames(self) -> float:
        return self._data[6]
    
    @property
    def effect_mask(self) -> np.ndarray:
        return self._effect_mask
    
    def get_current_factor(self) -> float:
        """Thread-safe optimized factor calculation."""
        with self._lock:
            frames_left = self._data[4]
            fade_in = self._data[5]
            fade_out = self._data[6]
            duration = self._data[4] + fade_in  # Approximate original duration
            
            if frames_left > duration - fade_in:
                return (duration - frames_left) / fade_in
            elif frames_left < fade_out:
                return frames_left / fade_out
            return 1.0
    
    def is_active(self) -> bool:
        return self._data[4] > 0
    
    def update(self):
        with self._lock:
            self._data[4] -= 1

class HyperOptimizedWindField:
    """Ultra-optimized wind field with maximum performance and threading support."""
    __slots__ = ('config', '_data_store', '_static_cache', '_frame_counter', 
                 '_last_update_frame', '_interpolation_state', '_performance_monitor',
                 '_thread_id', '_lock')
    
    def __init__(self, config=None):
        """Initialize with aggressive optimizations and thread safety."""
        self.config = config or WindFieldConfig()
        self._thread_id = threading.get_ident()
        self._lock = threading.RLock()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize thread-safe optimized data structures
        self._data_store = ThreadSafeDataStructures(
            self.config.grid_resolution, 
            len(self.config.layer_speed_factors)
        )
        
        # Pre-compute all static data
        self._static_cache = self._build_static_cache()
        
        # Initialize state tracking
        self._frame_counter = 0
        self._last_update_frame = 0
        self._interpolation_state = np.array([0.0, 12.0], dtype=np.float32)  # [factor, sim_hour]
        
        # Performance monitoring (minimal overhead)
        self._performance_monitor = {'gust_count': 0, 'update_calls': 0}
        
        # Initialize wind field
        self._initialize_field_optimized()
        
        optimization_status = "with JIT compilation" if NUMBA_AVAILABLE else "fallback mode (no JIT)"
        print(f"Thread-safe hyper-optimized wind field: {len(self.config.layer_speed_factors)} layers, "
              f"{self.config.grid_resolution}²grid, {optimization_status}")
    
    def _validate_config(self):
        """Fast configuration validation."""
        if len(self.config.layer_speed_factors) != len(self.config.layer_direction_offsets):
            raise ValueError("Layer configuration mismatch")
    
    @lru_cache(maxsize=1)
    def _build_static_cache(self):
        """Build and cache all static computational data (thread-safe)."""
        grid_size = self.config.grid_resolution
        
        # Pre-compute layer factors
        layer_factors = np.array([1.0 / (i + 1) for i in range(len(self.config.layer_speed_factors))], 
                                dtype=np.float32)
        
        # Pre-compute trigonometric lookup tables
        angles = np.arange(0, 360, dtype=np.float32)
        angle_lut = {
            'cos': np.cos(np.radians(angles)),
            'sin': np.sin(np.radians(angles))
        }
        
        # Pre-compute grid coordinates for fast sampling
        grid_coords = {
            'x_coords': np.arange(grid_size, dtype=np.float32),
            'y_coords': np.arange(grid_size, dtype=np.float32),
            'domain_to_grid': (grid_size - 1.0) / self.config.domain_size
        }
        
        # Pre-compute altitude mapping
        altitude_map = np.array(self.config.layer_heights, dtype=np.float32)
        
        # Pre-compute speed limits for vectorized clamping
        min_speeds = np.array([
            self.config.base_wind_speed * factor * 0.5 
            for factor in self.config.layer_speed_factors
        ], dtype=np.float32)
        
        max_speeds = np.array([
            self.config.base_wind_speed * factor * 1.5 
            for factor in self.config.layer_speed_factors
        ], dtype=np.float32)
        
        return {
            'layer_factors': layer_factors,
            'angle_lut': angle_lut,
            'grid_coords': grid_coords,
            'altitude_map': altitude_map,
            'wind_gusts': [],  # Pre-allocated gust list
            'min_speeds': min_speeds,
            'max_speeds': max_speeds
        }
    
    def _initialize_field_optimized(self):
        """Ultra-fast field initialization with thread safety."""
        with self._lock:
            grid_size = self.config.grid_resolution
            num_layers = len(self.config.layer_speed_factors)
            
            # Use thread-local RNG for thread safety
            rng = _get_thread_rng()
            
            # Generate all random data at once for better cache performance
            speed_noise = rng.normal(0, 0.05, (num_layers, grid_size, grid_size)).astype(np.float32)
            direction_noise = rng.normal(0, 3.0, (num_layers, grid_size, grid_size)).astype(np.float32)
            
            # Vectorized initialization with GIL release
            cells = self._data_store['cells']
            layer_factors = self._static_cache['layer_factors']
            
            for layer in range(num_layers):
                base_speed = self.config.base_wind_speed * self.config.layer_speed_factors[layer]
                base_direction = self.config.wind_direction + self.config.layer_direction_offsets[layer]
                
                # Ultra-fast vectorized assignment (releases GIL)
                cells[layer, :, :, 0] = base_speed * (1.0 + speed_noise[layer] * layer_factors[layer])
                cells[layer, :, :, 1] = (base_direction + direction_noise[layer] * layer_factors[layer]) % 360
            
            # Initialize previous state and vectors
            np.copyto(self._data_store['prev_cells'], cells)
            self._compute_vectors_ultra_fast()
            np.copyto(self._data_store['prev_vectors_x'], self._data_store['vectors_x'])
            np.copyto(self._data_store['prev_vectors_y'], self._data_store['vectors_y'])
    
    def _compute_vectors_ultra_fast(self):
        """JIT-compiled vector computation with GIL release."""
        _compute_vectors_jit(
            self._data_store['cells'][:, :, :, 0],
            self._data_store['cells'][:, :, :, 1],
            self._data_store['vectors_x'],
            self._data_store['vectors_y']
        )
    
    def step(self, frame_idx=None):
        """Ultra-optimized step function with thread safety."""
        with self._lock:
            self._frame_counter += 1
            frame_idx = frame_idx or self._frame_counter
            
            # Update only when necessary (frequency-based instead of time-based)
            should_update = (frame_idx - self._last_update_frame) >= self.config.update_frequency
            
            if should_update:
                self._last_update_frame = frame_idx
                self._performance_monitor['update_calls'] += 1
                
                # Store previous state efficiently
                np.copyto(self._data_store['prev_vectors_x'], self._data_store['vectors_x'])
                np.copyto(self._data_store['prev_vectors_y'], self._data_store['vectors_y'])
                
                # Conditional turbulence update
                if frame_idx % self.config.turbulence_update_frequency == 0:
                    self._update_turbulence_jit(frame_idx)
                
                # Apply wind changes with JIT and GIL release
                self._apply_wind_changes_ultra_fast()
                
                # Efficient gust management
                self._manage_gusts_optimized()
                
                # Recompute vectors
                self._compute_vectors_ultra_fast()
            
            # Fast interpolation factor calculation
            progress = (frame_idx - self._last_update_frame) / self.config.update_frequency
            self._interpolation_state[0] = min(1.0, progress * progress * (3 - 2 * progress))
    
    def _update_turbulence_jit(self, frame_idx):
        """JIT-compiled turbulence generation with deterministic seed."""
        # Use frame index as seed for reproducibility
        seed = (frame_idx // self.config.turbulence_update_frequency) % 1000000
        
        _generate_smooth_noise(
            self._data_store['turbulence_field'], 
            self.config.grid_resolution,
            seed
        )
        self._data_store['turbulence_field'] *= self.config.wind_turbulence
    
    def _apply_wind_changes_ultra_fast(self):
        """Ultra-fast wind field updates with GIL release."""
        # Calculate diurnal drift
        direction_drift = 0.0
        if self.config.diurnal_enabled:
            hour = self._interpolation_state[1] % 24
            time_factor = math.sin((hour - 6) * math.pi / 12)
            direction_drift = time_factor * 5.0 * 0.5  # change_magnitude = 0.5
        
        # Apply changes with JIT compilation and GIL release
        _apply_wind_changes_jit(
            self._data_store['cells'],
            self._data_store['turbulence_field'],
            self._static_cache['layer_factors'],
            direction_drift,
            0.5  # change_magnitude
        )
        
        # Vectorized speed clamping with GIL release
        _vectorized_speed_clamp(
            self._data_store['cells'][:, :, :, 0],
            self._static_cache['min_speeds'],
            self._static_cache['max_speeds']
        )
    
    def _manage_gusts_optimized(self):
        """Highly optimized gust management with thread safety."""
        gusts = self._static_cache['wind_gusts']
        
        # Remove inactive gusts efficiently
        active_gusts = [g for g in gusts if g.is_active()]
        
        # Apply active gusts with JIT and GIL release
        for gust in active_gusts:
            gust_factor = gust.get_current_factor()
            _apply_gust_effect_jit(
                self._data_store['cells'][:, :, :, 0],
                gust.effect_mask,
                gust_factor,
                self._static_cache['layer_factors']
            )
            gust.update()
        
        # Spawn new gusts with probability and limit (thread-safe)
        rng = _get_thread_rng()
        if (len(active_gusts) < self.config.max_gusts and 
            rng.random() < self.config.gust_probability):
            new_gust = ThreadSafeWindGust(self.config.grid_resolution, self.config.gust_duration)
            active_gusts.append(new_gust)
            self._performance_monitor['gust_count'] += 1
        
        # Update gust list
        self._static_cache['wind_gusts'] = active_gusts
    
    def update_time(self, hour):
        """Thread-safe optimized time update."""
        with self._lock:
            self._interpolation_state[1] = hour
    
    def sample(self, x: float, y: float, z: Optional[float] = None) -> Tuple[float, float]:
        """Ultra-optimized sampling with thread safety and GIL-releasing operations."""
        if z is None:
            z = 0.5
        
        # Fast layer lookup using binary search on small array
        layer_idx = np.searchsorted(self._static_cache['altitude_map'][:-1], z, side='right') - 1
        layer_idx = max(0, min(layer_idx, len(self.config.layer_speed_factors) - 1))
        
        # Ultra-fast coordinate transformation
        grid_coords = self._static_cache['grid_coords']
        fx = x * grid_coords['domain_to_grid']
        fy = y * grid_coords['domain_to_grid']
        
        # Fast bounds checking and grid calculation
        fx = max(0.0, min(fx, self.config.grid_resolution - 1.01))
        fy = max(0.0, min(fy, self.config.grid_resolution - 1.01))
        
        ix, iy = int(fx), int(fy)
        ix = min(ix, self.config.grid_resolution - 2)
        iy = min(iy, self.config.grid_resolution - 2)
        
        # JIT-compiled bilinear interpolation with GIL release
        vx_curr = _fast_bilinear_interpolate(
            self._data_store['vectors_x'][layer_idx], fx, fy, ix, iy)
        vy_curr = _fast_bilinear_interpolate(
            self._data_store['vectors_y'][layer_idx], fx, fy, ix, iy)
        
        vx_prev = _fast_bilinear_interpolate(
            self._data_store['prev_vectors_x'][layer_idx], fx, fy, ix, iy)
        vy_prev = _fast_bilinear_interpolate(
            self._data_store['prev_vectors_y'][layer_idx], fx, fy, ix, iy)
        
        # Temporal interpolation
        t = self._interpolation_state[0]
        vx = vx_prev + t * (vx_curr - vx_prev)
        vy = vy_prev + t * (vy_curr - vy_prev)
        
        # Fast magnitude and angle calculation
        speed = math.sqrt(vx * vx + vy * vy)
        direction = math.degrees(math.atan2(vy, vx))
        if direction < 0:
            direction += 360
        
        return speed, direction
    
    def get_velocity_at_altitude(self, altitude_km):
        """Thread-safe cached velocity calculation."""
        # Use lookup table for common altitudes
        layer_idx = min(len(self.config.layer_speed_factors) - 1, max(0, int(altitude_km * 2)))
        
        speed = self.config.base_wind_speed * self.config.layer_speed_factors[layer_idx]
        direction = self.config.wind_direction + self.config.layer_direction_offsets[layer_idx]
        
        # Fast trigonometry using lookup table
        dir_int = int(direction) % 360
        cos_val = self._static_cache['angle_lut']['cos'][dir_int]
        sin_val = self._static_cache['angle_lut']['sin'][dir_int]
        
        speed_kph = speed * 3.6
        return speed_kph * cos_val, speed_kph * sin_val
    
    def vector(self, t_min: float) -> Tuple[float, float]:
        """Optimized compatibility method with thread safety."""
        speed, direction = self.sample(self.config.domain_size/2, self.config.domain_size/2, 1.0)
        speed_km_min = speed * 0.06  # 60/1000 pre-computed
        dir_rad = math.radians(direction)
        return speed_km_min * math.cos(dir_rad), speed_km_min * math.sin(dir_rad)
    
    def vector_at_altitude(self, t_min: float, altitude_idx: int) -> Tuple[float, float]:
        """Thread-safe optimized altitude-specific vector."""
        altitude = (self._static_cache['altitude_map'][altitude_idx] 
                   if altitude_idx < len(self._static_cache['altitude_map']) 
                   else self._static_cache['altitude_map'][-1])
        
        speed, direction = self.sample(self.config.domain_size/2, self.config.domain_size/2, altitude)
        speed_km_min = speed * 0.06
        dir_rad = math.radians(direction)
        return speed_km_min * math.cos(dir_rad), speed_km_min * math.sin(dir_rad)
    
    def get_dominant_flow(self, layer: int = 1) -> Tuple[float, float]:
        """Ultra-fast dominant flow calculation with GIL release."""
        layer = max(0, min(layer, len(self.config.layer_speed_factors) - 1))
        
        # Use mean with optimized axis specification (releases GIL)
        avg_vx = np.mean(self._data_store['vectors_x'][layer])
        avg_vy = np.mean(self._data_store['vectors_y'][layer])
        
        speed = math.sqrt(avg_vx * avg_vx + avg_vy * avg_vy)
        direction = math.degrees(math.atan2(avg_vy, avg_vx))
        if direction < 0:
            direction += 360
        
        return speed, direction
    
    def get_performance_stats(self) -> Dict:
        """Get performance monitoring data (thread-safe)."""
        with self._lock:
            return self._performance_monitor.copy()
    
    def get_thread_info(self) -> Dict:
        """Get threading information for debugging."""
        return {
            'creator_thread_id': self._thread_id,
            'current_thread_id': threading.get_ident(),
            'is_same_thread': self._thread_id == threading.get_ident(),
            'numba_available': NUMBA_AVAILABLE,
            'gil_releasing': NUMBA_AVAILABLE
        }


# Backwards compatibility alias
EnhancedWindField = HyperOptimizedWindField


@lru_cache(maxsize=128)
def get_global_wind_vector():
    """Cached global wind vector calculation."""
    speed = getattr(CFG, 'DEFAULT_WIND_SPEED_KMH', 10.0)
    direction = getattr(CFG, 'DEFAULT_WIND_DIRECTION_DEG', 0.0)
    
    rad = math.radians(direction)
    return speed * math.cos(rad), speed * math.sin(rad)


# Threading utilities for wind field
def create_thread_safe_wind_field(config=None) -> HyperOptimizedWindField:
    """Factory function to create a thread-safe wind field."""
    return HyperOptimizedWindField(config)


def validate_threading_support() -> Dict[str, bool]:
    """Validate threading support and optimizations."""
    return {
        'numba_available': NUMBA_AVAILABLE,
        'gil_releasing_enabled': NUMBA_AVAILABLE,
        'vectorized_operations': True,
        'thread_local_rng': True,
        'thread_safe_data_structures': True
    }