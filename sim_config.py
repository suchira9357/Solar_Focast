# sim_config.py - Threading-Enhanced Configuration
"""
Simulation configuration variables with comprehensive threading support
Supports both single-threaded and multi-threaded execution modes
"""

import os
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# ===== BASIC SIMULATION PARAMETERS =====
SIMULATION_NAME = "Solar Farm Cloud Simulation"
SIMULATION_VERSION = "2.0-Threading"
FPS = 60
AREA_SIZE_KM = 50
DOMAIN_SIZE_M = 10000
BASE_WIND_SPEED = 2.0
MOVEMENT_MULTIPLIER = 1.0
SINGLE_CLOUD_MODE = True
SPAWN_PROBABILITY = 1.0
WIND_GRID = 100  # Default grid resolution, adjust as needed
BASE_WIND_DIRECTION = 0.0  # Default wind direction in degrees (0 = east, 90 = north)

# Cloud movement direction (135° = NE→SW movement)
CLOUD_DIRECTION = 135.0

# Additional cloud configuration
MAX_PARCELS = 6  # Maximum number of clouds
FORCE_INITIAL_CLOUD = True  # Force spawn at least one cloud
CLOUD_GROWTH_FRAMES = 60
CLOUD_STABLE_FRAMES = 1800
CLOUD_DECAY_FRAMES = 300

# Cloud appearance settings
CLOUD_TRANSMITTANCE = 0.5  # 0 = fully opaque, 1 = fully transparent
SHADOW_FADE_MS = 500  # Default fade duration in milliseconds

# Physics settings
PHYSICS_TIMESTEP = 1.0 / 60.0  # Physics timestep in seconds (60 FPS)
MIN_TIMESTEP = 1.0 / 240.0  # Minimum timestep for stability
MAX_TIMESTEP = 1.0 / 15.0   # Maximum timestep to prevent large jumps

# ===== THREADING CONFIGURATION =====

class ThreadingMode(Enum):
    """Threading execution modes."""
    DISABLED = "disabled"      # Single-threaded execution
    WORKER = "worker"         # Dedicated physics worker thread
    AUTO = "auto"             # Automatically choose based on CPU count

# Core threading settings
THREADING_ENABLED = True  # Global threading enable/disable
THREADING_MODE = ThreadingMode.WORKER  # Default threading mode
THREADING_AUTO_DETECT = True  # Auto-detect optimal threading settings

# Worker thread configuration
WORKER_THREAD_PRIORITY = threading.THREAD_PRIORITY_NORMAL if hasattr(threading, 'THREAD_PRIORITY_NORMAL') else 0
WORKER_THREAD_NAME = "PhysicsWorker"
WORKER_THREAD_DAEMON = True  # Worker thread dies with main thread

# Queue configuration (as specified in requirements)
DT_QUEUE_SIZE = 5           # main → worker: delta time values (≤ 5)
RESULT_QUEUE_SIZE = 3       # worker → main: simulation results (≤ 3)
QUEUE_TIMEOUT_SECONDS = 0.1  # Timeout for queue operations
QUEUE_BLOCK_ON_FULL = False  # Drop frames instead of blocking

# Worker performance settings
WORKER_UPDATE_FREQUENCY = 60  # Target worker FPS
WORKER_MAX_CONSECUTIVE_ERRORS = 5  # Max errors before reset
WORKER_ERROR_BACKOFF_TIME = 0.1  # Seconds to wait after error
WORKER_RESTART_ATTEMPTS = 3  # Max worker restart attempts
WORKER_RESTART_COOLDOWN = 2.0  # Seconds between restart attempts

# Threading performance tuning
THREAD_AFFINITY_ENABLED = False  # Enable CPU affinity (requires psutil)
THREAD_AFFINITY_MAIN = None      # Main thread CPU affinity
THREAD_AFFINITY_WORKER = None    # Worker thread CPU affinity
GIL_RELEASE_THRESHOLD = 100      # Minimum array size for GIL release optimizations

# ===== MEMORY MANAGEMENT =====

# Memory usage limits
MAX_QUEUE_MEMORY_MB = 50     # Maximum memory for all queues
MAX_CACHE_SIZE = 1000        # Maximum cache entries
MEMORY_CHECK_INTERVAL = 10.0  # Seconds between memory checks
GARBAGE_COLLECTION_INTERVAL = 30.0  # Seconds between GC calls

# Data serialization settings
SERIALIZE_NUMPY_ARRAYS = True    # Serialize NumPy arrays efficiently
COMPRESS_QUEUE_DATA = False      # Compress data in queues (CPU vs memory tradeoff)
MAX_SERIALIZED_SIZE_KB = 100     # Maximum size for serialized objects

# ===== ERROR HANDLING & RECOVERY =====

# Error handling configuration
ENABLE_ERROR_RECOVERY = True     # Enable automatic error recovery
LOG_WORKER_ERRORS = True         # Log worker thread errors
FALLBACK_MODE_ENABLED = True     # Enable fallback simulation mode
FALLBACK_TIMEOUT_SECONDS = 5.0   # Timeout before entering fallback mode

# Graceful shutdown settings
SHUTDOWN_TIMEOUT_SECONDS = 3.0   # Max time to wait for worker shutdown
FORCE_SHUTDOWN_ON_TIMEOUT = True # Force shutdown if timeout exceeded
CLEANUP_ON_EXIT = True           # Clean up resources on exit

# ===== DEBUGGING & MONITORING =====

# Debug configuration
DEBUG_THREADING = False          # Enable threading debug output
DEBUG_QUEUE_OPERATIONS = False   # Log queue operations
DEBUG_PERFORMANCE = False        # Enable performance debugging
DEBUG_MEMORY_USAGE = False       # Monitor memory usage
DEBUG_WORKER_LIFECYCLE = True    # Log worker start/stop events

# Performance monitoring
ENABLE_PERFORMANCE_MONITORING = True  # Enable performance stats collection
PERFORMANCE_SAMPLE_SIZE = 100         # Number of samples for rolling averages
PERFORMANCE_REPORT_INTERVAL = 5.0     # Seconds between performance reports
MONITOR_QUEUE_HEALTH = True           # Monitor queue health metrics

# Profiling support
ENABLE_PROFILING = False         # Enable code profiling
PROFILING_OUTPUT_DIR = "profiles"  # Directory for profiling output
PROFILE_WORKER_THREAD = False    # Profile worker thread performance

# ===== FALLBACK SIMULATION PARAMETERS =====

# Fallback mode configuration (when worker thread fails)
FALLBACK_CLOUD_COUNT = 1         # Number of clouds in fallback mode
FALLBACK_CLOUD_SPEED = 5.0       # Fallback cloud speed (km/h)
FALLBACK_CLOUD_DIRECTION = 135.0 # Fallback cloud direction
FALLBACK_CLOUD_OPACITY = 0.5     # Fallback cloud opacity
FALLBACK_POWER_LEVEL = 0.8       # Fallback power output level (80%)
FALLBACK_UPDATE_RATE = 30        # Fallback update rate (FPS)

# ===== CLOUD TYPE CONFIGURATION =====

# Cloud type weights for spawning
CLOUD_TYPE_WEIGHTS = {
    "cumulus": 0.5,
    "cirrus": 0.3,
    "cumulonimbus": 0.2
}

# Cloud type properties
CLOUD_TYPES = {
    "cumulus": {
        "alt_km": 1.5,
        "r_km": (1.0, 3.0),
        "opacity_max": 0.8,
        "speed_k": 1.0
    },
    "cirrus": {
        "alt_km": 8.0,
        "r_km": (2.0, 5.0),
        "opacity_max": 0.6,
        "speed_k": 1.2
    },
    "cumulonimbus": {
        "alt_km": 2.0,
        "r_km": (2.0, 6.0),
        "opacity_max": 0.9,
        "speed_k": 0.8
    }
}

# ===== ADVANCED THREADING FEATURES =====

# Thread synchronization
USE_THREAD_BARRIERS = False      # Use threading barriers for sync
BARRIER_TIMEOUT = 1.0            # Barrier timeout in seconds
THREAD_POOL_SIZE = 2             # Size of thread pool for batch operations

# Lock configuration
LOCK_TIMEOUT_SECONDS = 1.0       # Timeout for acquiring locks
USE_RLOCK = True                 # Use reentrant locks where possible
DEADLOCK_DETECTION = True        # Enable deadlock detection

# Advanced queue features
PRIORITY_QUEUE_ENABLED = False   # Use priority queues
QUEUE_PERSISTENCE = False        # Persist queue contents to disk
QUEUE_COMPRESSION = False        # Compress queue data

# ===== PLATFORM-SPECIFIC SETTINGS =====

# Platform optimizations
PLATFORM_OPTIMIZATIONS = True    # Enable platform-specific optimizations
WINDOWS_HIGH_RESOLUTION_TIMER = True  # Use high-res timer on Windows
LINUX_REALTIME_PRIORITY = False      # Use real-time priority on Linux
MACOS_GRAND_CENTRAL_DISPATCH = False # Use GCD on macOS

# ===== CONFIGURATION VALIDATION =====

@dataclass
class ThreadingConfig:
    """Immutable threading configuration with validation."""
    enabled: bool = THREADING_ENABLED
    mode: ThreadingMode = THREADING_MODE
    dt_queue_size: int = DT_QUEUE_SIZE
    result_queue_size: int = RESULT_QUEUE_SIZE
    worker_fps: int = WORKER_UPDATE_FREQUENCY
    max_errors: int = WORKER_MAX_CONSECUTIVE_ERRORS
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.dt_queue_size < 1 or self.dt_queue_size > 10:
            raise ValueError(f"DT queue size must be 1-10, got {self.dt_queue_size}")
        if self.result_queue_size < 1 or self.result_queue_size > 5:
            raise ValueError(f"Result queue size must be 1-5, got {self.result_queue_size}")
        if self.worker_fps < 1 or self.worker_fps > 240:
            raise ValueError(f"Worker FPS must be 1-240, got {self.worker_fps}")

def validate_threading_config() -> bool:
    """Validate threading configuration and return True if valid."""
    try:
        config = ThreadingConfig()
        
        # Check for conflicting settings
        if not THREADING_ENABLED and THREADING_MODE != ThreadingMode.DISABLED:
            print("Warning: Threading disabled but mode is not DISABLED")
            return False
        
        # Validate queue sizes (as per requirements)
        if DT_QUEUE_SIZE > 5:
            print(f"Error: DT_QUEUE_SIZE ({DT_QUEUE_SIZE}) exceeds requirement of ≤ 5")
            return False
        
        if RESULT_QUEUE_SIZE > 3:
            print(f"Error: RESULT_QUEUE_SIZE ({RESULT_QUEUE_SIZE}) exceeds requirement of ≤ 3")
            return False
        
        print("Threading configuration validation passed")
        return True
        
    except Exception as e:
        print(f"Threading configuration validation failed: {e}")
        return False

# ===== RUNTIME CONFIGURATION =====

class RuntimeConfig:
    """Runtime configuration manager with thread-safe updates."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._config = {}
        self._callbacks = []
    
    def set(self, key: str, value: Any) -> None:
        """Thread-safe configuration update."""
        with self._lock:
            old_value = self._config.get(key)
            self._config[key] = value
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(key, old_value, value)
                except Exception as e:
                    print(f"Configuration callback error: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe configuration retrieval."""
        with self._lock:
            return self._config.get(key, default)
    
    def add_callback(self, callback):
        """Add configuration change callback."""
        with self._lock:
            self._callbacks.append(callback)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        with self._lock:
            return self._config.copy()

# Global runtime configuration instance
runtime_config = RuntimeConfig()

# ===== ENVIRONMENT-BASED OVERRIDES =====

def apply_environment_overrides():
    """Apply configuration overrides from environment variables."""
    env_overrides = {
        'SOLAR_SIM_THREADING': ('THREADING_ENABLED', bool),
        'SOLAR_SIM_DEBUG': ('DEBUG_THREADING', bool),
        'SOLAR_SIM_FPS': ('FPS', int),
        'SOLAR_SIM_WORKER_FPS': ('WORKER_UPDATE_FREQUENCY', int),
        'SOLAR_SIM_DT_QUEUE_SIZE': ('DT_QUEUE_SIZE', int),
        'SOLAR_SIM_RESULT_QUEUE_SIZE': ('RESULT_QUEUE_SIZE', int),
        'SOLAR_SIM_NO_THREADING': ('THREADING_ENABLED', lambda x: not bool(x)),
    }
    
    for env_var, (config_var, converter) in env_overrides.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            try:
                if converter == bool:
                    value = env_value.lower() in ('true', '1', 'yes', 'on')
                elif callable(converter):
                    value = converter(env_value)
                else:
                    value = converter(env_value)
                
                globals()[config_var] = value
                print(f"Environment override: {config_var} = {value}")
                
            except (ValueError, TypeError) as e:
                print(f"Invalid environment value for {env_var}: {env_value} ({e})")

# ===== CONFIGURATION PRESETS =====

THREADING_PRESETS = {
    'high_performance': {
        'THREADING_ENABLED': True,
        'THREADING_MODE': ThreadingMode.WORKER,
        'DT_QUEUE_SIZE': 5,
        'RESULT_QUEUE_SIZE': 3,
        'WORKER_UPDATE_FREQUENCY': 120,
        'ENABLE_PERFORMANCE_MONITORING': True,
        'GIL_RELEASE_THRESHOLD': 50,
    },
    'stable': {
        'THREADING_ENABLED': True,
        'THREADING_MODE': ThreadingMode.WORKER,
        'DT_QUEUE_SIZE': 3,
        'RESULT_QUEUE_SIZE': 2,
        'WORKER_UPDATE_FREQUENCY': 60,
        'WORKER_MAX_CONSECUTIVE_ERRORS': 3,
        'ENABLE_ERROR_RECOVERY': True,
    },
    'debug': {
        'THREADING_ENABLED': True,
        'THREADING_MODE': ThreadingMode.WORKER,
        'DT_QUEUE_SIZE': 2,
        'RESULT_QUEUE_SIZE': 1,
        'DEBUG_THREADING': True,
        'DEBUG_QUEUE_OPERATIONS': True,
        'DEBUG_PERFORMANCE': True,
        'LOG_WORKER_ERRORS': True,
    },
    'single_threaded': {
        'THREADING_ENABLED': False,
        'THREADING_MODE': ThreadingMode.DISABLED,
        'FALLBACK_MODE_ENABLED': False,
        'DEBUG_THREADING': False,
    }
}

def apply_preset(preset_name: str) -> bool:
    """Apply a configuration preset."""
    if preset_name not in THREADING_PRESETS:
        print(f"Unknown preset: {preset_name}")
        return False
    
    preset = THREADING_PRESETS[preset_name]
    for key, value in preset.items():
        if key in globals():
            globals()[key] = value
            print(f"Preset {preset_name}: {key} = {value}")
        else:
            print(f"Warning: Unknown config key in preset: {key}")
    
    return True

# ===== INITIALIZATION =====

def initialize_config():
    """Initialize configuration system."""
    try:
        # Apply environment overrides
        apply_environment_overrides()
        
        # Validate configuration
        if not validate_threading_config():
            print("Warning: Invalid threading configuration detected")
        
        # Set derived values
        global PHYSICS_TIMESTEP
        if WORKER_UPDATE_FREQUENCY > 0:
            PHYSICS_TIMESTEP = 1.0 / WORKER_UPDATE_FREQUENCY
        
        print(f"Configuration initialized - Threading: {THREADING_ENABLED}, "
              f"Mode: {THREADING_MODE.value}, FPS: {FPS}")
        
        return True
        
    except Exception as e:
        print(f"Configuration initialization failed: {e}")
        return False

# ===== COMPATIBILITY FUNCTIONS =====

def is_threading_enabled() -> bool:
    """Check if threading is enabled."""
    return THREADING_ENABLED and THREADING_MODE != ThreadingMode.DISABLED

def get_queue_sizes() -> tuple[int, int]:
    """Get (dt_queue_size, result_queue_size) tuple."""
    return (DT_QUEUE_SIZE, RESULT_QUEUE_SIZE)

def get_worker_config() -> dict:
    """Get worker thread configuration."""
    return {
        'fps': WORKER_UPDATE_FREQUENCY,
        'max_errors': WORKER_MAX_CONSECUTIVE_ERRORS,
        'error_backoff': WORKER_ERROR_BACKOFF_TIME,
        'restart_attempts': WORKER_RESTART_ATTEMPTS,
        'restart_cooldown': WORKER_RESTART_COOLDOWN,
        'thread_name': WORKER_THREAD_NAME,
        'daemon': WORKER_THREAD_DAEMON,
    }

def get_performance_config() -> dict:
    """Get performance monitoring configuration."""
    return {
        'enabled': ENABLE_PERFORMANCE_MONITORING,
        'sample_size': PERFORMANCE_SAMPLE_SIZE,
        'report_interval': PERFORMANCE_REPORT_INTERVAL,
        'monitor_queues': MONITOR_QUEUE_HEALTH,
        'debug_performance': DEBUG_PERFORMANCE,
    }

# ===== BACKWARDS COMPATIBILITY =====

# Ensure all original variables are still available
DEFAULT_WIND_SPEED_KMH = BASE_WIND_SPEED * 3.6  # Convert m/s to km/h
DEFAULT_WIND_DIRECTION_DEG = BASE_WIND_DIRECTION
TIMESTEP_MINUTES = 5  # For compatibility with existing code
TIME_SCALE = 1.0      # Time scale factor

# Initialize configuration on module import
_config_initialized = initialize_config()

if not _config_initialized:
    print("Warning: Configuration initialization failed, using defaults")

# Export key functions and classes for external use
__all__ = [
    'ThreadingMode', 'ThreadingConfig', 'RuntimeConfig',
    'validate_threading_config', 'apply_preset', 'is_threading_enabled',
    'get_queue_sizes', 'get_worker_config', 'get_performance_config',
    'runtime_config'
]