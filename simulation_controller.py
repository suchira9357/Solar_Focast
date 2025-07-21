#!/usr/bin/env python3
"""
Simulation Controller - Complete Threading Implementation
Orchestrates weather, power, shadows with background physics thread support

Threading Architecture:
┌─────────────────┐    dt     ┌─────────────────┐
│   Main Thread   │─────────→│  Physics Thread │
│ (Rendering & I/O)│          │   (Worker)      │
└─────────────────┘←─────────┐                 │
         │  result           │                 │
         └───────────────────┘─────────────────┘
"""
import threading
import traceback
import logging
import os
import json
import copy
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque
from dataclasses import dataclass, field
from contextlib import contextmanager

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import configuration
try:
    import sim_config as CFG
except ImportError:
    # Fallback configuration
    class CFG:
        PHYSICS_TIMESTEP = 1/60.0
        DOMAIN_SIZE_M = 10000
        AREA_SIZE_KM = 10
        TIME_SCALE = 1.0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking for simulation controller."""
    frames_processed: int = 0
    total_processing_time: float = 0.0
    avg_frame_time_ms: float = 0.0
    max_frame_time_ms: float = 0.0
    min_frame_time_ms: float = float('inf')
    
    # Threading-specific metrics
    worker_dt_queue_size: int = 0
    worker_result_queue_size: int = 0
    worker_queue_pressure: float = 0.0
    worker_processing_time_ms: float = 0.0
    worker_error_count: int = 0
    worker_recovery_count: int = 0
    
    # Physics metrics
    cloud_count: int = 0
    active_panels: int = 0
    total_power_kw: float = 0.0
    cloud_coverage_pct: float = 0.0
    wind_speed_ms: float = 0.0
    
    last_update_time: float = 0.0
    
    def update_frame_timing(self, frame_time_ms: float):
        """Update frame timing statistics."""
        self.frames_processed += 1
        self.total_processing_time += frame_time_ms / 1000.0
        
        if frame_time_ms > self.max_frame_time_ms:
            self.max_frame_time_ms = frame_time_ms
        if frame_time_ms < self.min_frame_time_ms:
            self.min_frame_time_ms = frame_time_ms
            
        self.avg_frame_time_ms = (self.total_processing_time / self.frames_processed) * 1000.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'frames_processed': self.frames_processed,
            'avg_frame_time_ms': self.avg_frame_time_ms,
            'max_frame_time_ms': self.max_frame_time_ms,
            'min_frame_time_ms': self.min_frame_time_ms,
            'worker_dt_queue_size': self.worker_dt_queue_size,
            'worker_result_queue_size': self.worker_result_queue_size,
            'worker_queue_pressure': self.worker_queue_pressure,
            'worker_processing_time_ms': self.worker_processing_time_ms,
            'worker_error_count': self.worker_error_count,
            'worker_recovery_count': self.worker_recovery_count,
            'cloud_count': self.cloud_count,
            'active_panels': self.active_panels,
            'total_power_kw': self.total_power_kw,
            'cloud_coverage_pct': self.cloud_coverage_pct,
            'wind_speed_ms': self.wind_speed_ms
        }


@dataclass
class SimulationConfig:
    """Configuration for simulation controller with threading support."""
    use_threading: bool = True
    worker_queue_sizes: Dict[str, int] = field(default_factory=lambda: {'dt': 5, 'result': 3})
    target_fps: float = 60.0
    physics_timestep: float = 1/60.0
    time_scale: float = 1.0
    start_time: float = 11.0
    timestep_minutes: int = 5
    enable_performance_monitoring: bool = True
    performance_update_interval: float = 1.0
    coordinates_file: str = "extended_coordinates.csv"
    validate_coordinates: bool = True
    weather_seed: Optional[int] = None
    latitude: float = 6.9271
    longitude: float = 79.8612
    timezone_offset: float = 5.5
    debug_mode: bool = False
    verbose_logging: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SimulationConfig':
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'use_threading': self.use_threading,
            'worker_queue_sizes': self.worker_queue_sizes,
            'target_fps': self.target_fps,
            'physics_timestep': self.physics_timestep,
            'time_scale': self.time_scale,
            'start_time': self.start_time,
            'timestep_minutes': self.timestep_minutes,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'performance_update_interval': self.performance_update_interval,
            'coordinates_file': self.coordinates_file,
            'validate_coordinates': self.validate_coordinates,
            'weather_seed': self.weather_seed,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'timezone_offset': self.timezone_offset,
            'debug_mode': self.debug_mode,
            'verbose_logging': self.verbose_logging
        }
    
    def save_to_file(self, config_path: str):
        try:
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Could not save config to {config_path}: {e}")


class SimulationController:
    """
    Enhanced simulation controller with complete threading support.
    
    Threading Architecture:
    ┌─────────────────┐    dt     ┌─────────────────┐
    │   Main Thread   │─────────→│  Physics Thread │
    │ (Rendering & I/O)│          │   (Worker)      │
    └─────────────────┘←─────────┐                 │
             │  result           │                 │
             └───────────────────┘─────────────────┘
    """
    
    def __init__(self, config: Optional[Union[SimulationConfig, Dict[str, Any]]] = None,
                 panel_df: Optional[pd.DataFrame] = None,
                 **kwargs):
        """
        Initialize simulation controller with threading support.
        
        Args:
            config: SimulationConfig instance or dict
            panel_df: Panel DataFrame with coordinates and properties
            **kwargs: Additional configuration overrides
        """
        # Initialize configuration
        if isinstance(config, dict):
            self.config = SimulationConfig.from_dict(config)
        elif isinstance(config, SimulationConfig):
            self.config = config
        else:
            self.config = SimulationConfig()
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Validation
        self._validate_configuration()
        
        if self.config.verbose_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"Initializing SimulationController:")
        logger.info(f"  Threading: {'enabled' if self.config.use_threading else 'disabled'}")
        logger.info(f"  Start time: {self.config.start_time:.1f}h")
        logger.info(f"  Debug mode: {self.config.debug_mode}")
        
        # State management
        self.current_hour = self.config.start_time
        self.frame_count = 0
        self.simulation_start_time = None  # Will be set when simulation starts
        self.last_update_time = None
        
        # Load panel data
        self.panel_df = self._load_panel_data(panel_df)
        
        # Threading components
        self.worker = None
        self._last_state = None
        self._simulation_lock = threading.RLock()
        self._shutdown_requested = False
        self._context_managed = False
        
        # Performance monitoring
        self.performance_metrics = PerformanceMetrics()
        self.performance_history = deque(maxlen=100)
        self.last_performance_update = 0.0
        
        # Initialize simulation components
        self._initialize_components()
        
        # Initialize physics worker if threading is enabled
        if self.config.use_threading:
            self._initialize_physics_worker()
        
        logger.info("✓ SimulationController initialization complete")
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        if self.config.physics_timestep <= 0:
            raise ValueError("physics_timestep must be positive")
        if self.config.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if not (0 <= self.config.start_time < 24):
            logger.warning(f"start_time {self.config.start_time} outside 0-24 range")
            self.config.start_time = self.config.start_time % 24.0
    
    def _load_panel_data(self, panel_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Load or validate panel data."""
        if panel_df is not None:
            logger.info(f"Using provided panel data: {len(panel_df)} panels")
            return panel_df.copy()
        
        # Try to load from coordinates file
        if os.path.exists(self.config.coordinates_file):
            try:
                panel_df = pd.read_csv(self.config.coordinates_file)
                logger.info(f"Loaded panel data from {self.config.coordinates_file}: {len(panel_df)} panels")
                
                # Validate required columns
                required_cols = ['panel_id', 'x_km', 'y_km', 'x', 'y']
                missing_cols = [col for col in required_cols if col not in panel_df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in panel data: {missing_cols}")
                
                # Add default power capacity if missing
                if 'power_capacity' not in panel_df.columns:
                    panel_df = panel_df.copy()
                    panel_df['power_capacity'] = 5.0  # 5kW default
                    logger.info("Added default power_capacity (5.0 kW) for all panels")
                
                return panel_df
                
            except Exception as e:
                logger.error(f"Error loading panel data from {self.config.coordinates_file}: {e}")
        
        # Create default panel data
        logger.info("Creating default panel data")
        return self._create_default_panel_data()
    
    def _create_default_panel_data(self) -> pd.DataFrame:
        """Create default panel data for testing."""
        num_panels = 36
        grid_size = int(np.sqrt(num_panels))
        
        panel_data = []
        for i in range(grid_size):
            for j in range(grid_size):
                if i * grid_size + j < num_panels:
                    panel_data.append({
                        'panel_id': f"P{i*grid_size+j+1:03d}",
                        'x_km': 1.0 + i * 1.5,
                        'y_km': 1.0 + j * 1.5,
                        'x': 100 + i * 150,
                        'y': 100 + j * 150,
                        'power_capacity': 5.0  # 5kW panels
                    })
        
        logger.info(f"Generated {len(panel_data)} default panels")
        return pd.DataFrame(panel_data)
    
    def _initialize_components(self):
        """Initialize simulation components."""
        try:
            # Weather system
            from cloud_simulation import OptimizedWeatherSystem
            seed = self.config.weather_seed or hash(str(self.config.start_time)) % 1000000
            self.weather_system = OptimizedWeatherSystem(seed=seed)
            logger.info("✓ Weather system initialized")
            
            # Wind field
            try:
                from enhanced_wind_field import HyperOptimizedWindField
                self.wind_field = HyperOptimizedWindField()
                logger.info("✓ Wind field initialized")
            except ImportError:
                logger.warning("Enhanced wind field not available, using fallback")
                self.wind_field = None
            
            # Shadow calculator
            from shadow_calculator import ShadowCalculator
            self.shadow_calculator = ShadowCalculator(
                panel_df=self.panel_df,
                domain_size=CFG.DOMAIN_SIZE_M
            )
            logger.info("✓ Shadow calculator initialized")
            
            # Power simulator
            from power_simulator import PowerSimulator
            self.power_simulator = PowerSimulator(
                panel_df=self.panel_df,
                latitude=self.config.latitude,
                longitude=self.config.longitude,
                timezone_offset=self.config.timezone_offset
            )
            logger.info("✓ Power simulator initialized")
            
            # Apply global time scale
            CFG.TIME_SCALE = self.config.time_scale
            CFG.PHYSICS_TIMESTEP = self.config.physics_timestep
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            if self.config.debug_mode:
                traceback.print_exc()
            raise
    
    def _initialize_physics_worker(self):
        """Initialize physics worker thread."""
        try:
            logger.debug("Initializing physics worker...")
            from worker_thread import PhysicsWorker
            self.worker = PhysicsWorker(
                weather_system=self.weather_system,
                shadow_calculator=self.shadow_calculator,
                power_simulator=self.power_simulator,
                panel_df=self.panel_df
            )
            self.worker.start()
            logger.info("✓ Physics worker thread started")
        except Exception as e:
            logger.error(f"Error initializing physics worker: {e}")
            logger.warning("Falling back to non-threaded mode")
            self.config.use_threading = False
    
    @contextmanager
    def _simulation_context(self):
        """Context manager for simulation state."""
        if self._context_managed:
            yield
        else:
            self._context_managed = True
            try:
                if self.simulation_start_time is None:
                    self.simulation_start_time = 0.0  # Start time reference
                    self.last_update_time = 0.0
                yield
            finally:
                self._context_managed = False
    
    def step(self, dt: float = None) -> Dict[str, Any]:
        """
        Main simulation step - handles both threaded and non-threaded modes.
        
        Args:
            dt: Delta time in seconds (defaults to physics timestep)
            
        Returns:
            Dictionary containing simulation state
        """
        if dt is None:
            dt = self.config.physics_timestep
        
        with self._simulation_context():
            frame_start_time = self._get_current_time()
            
            try:
                if self.config.use_threading and self.worker:
                    result = self._step_threaded(dt)
                else:
                    result = self._step_direct(dt)
                
                # Update performance metrics
                frame_time_ms = (self._get_current_time() - frame_start_time) * 1000.0
                self._update_performance_metrics(frame_time_ms, result)
                
                # Update frame count and timing
                self.frame_count += 1
                if self.last_update_time is not None:
                    self.current_hour += dt * self.config.time_scale / 3600.0
                    self.current_hour = self.current_hour % 24.0
                self.last_update_time = frame_start_time
                
                return result
                
            except Exception as e:
                logger.error(f"Error in simulation step: {e}")
                if self.config.debug_mode:
                    traceback.print_exc()
                
                # Return fallback state
                return self._create_fallback_state()
    
    def _get_current_time(self) -> float:
        """Get current time for performance measurement."""
        return self.frame_count * self.config.physics_timestep  # Simulation time
    
    def _step_threaded(self, dt: float) -> Dict[str, Any]:
        """Handle threaded simulation step."""
        # Submit timestep to worker
        if self.worker:
            success = self.worker.submit_timestep(dt)
            if not success and self.config.debug_mode:
                logger.debug("Worker queue full, dropping timestep")
        
        # Get latest results from worker
        latest_state = None
        if self.worker:
            latest_state = self.worker.get_latest_results()
        
        if latest_state:
            # Convert worker state to simulation result
            result = latest_state.to_dict()
            self._last_state = result
            return result
        else:
            # No new results, use cached state
            if self._last_state:
                return self._last_state
            else:
                return self._create_fallback_state()
    
    def _step_direct(self, dt: float) -> Dict[str, Any]:
        """
        Handle direct (non-threaded) simulation step.
        
        This method contains the CRITICAL FIX for the t_s parameter issue.
        """
        # Update simulation time
        sim_time = self.frame_count * dt
        
        # CRITICAL FIX: Update weather system - removed t_s parameter
        if self.weather_system:
            try:
                if hasattr(self.weather_system, 'step'):
                    # FIXED: Only pass dt, not t_s
                    self.weather_system.step(dt=dt)
                elif hasattr(self.weather_system, 'update'):
                    self.weather_system.update(dt, current_hour=self.current_hour)
            except Exception as e:
                logger.warning(f"Error updating weather system: {e}")
        
        # Get cloud data
        cloud_ellipses = []
        cloud_positions = []
        cloud_coverage = 0.0
        wind_speed = 0.0
        wind_direction = 0.0
        
        if self.weather_system:
            try:
                # Get cloud ellipses
                if hasattr(self.weather_system, 'parcels'):
                    try:
                        from cloud_simulation import collect_visible_ellipses
                        cloud_ellipses = collect_visible_ellipses(self.weather_system.parcels)
                    except ImportError:
                        # Fallback: create basic ellipses from parcels
                        cloud_ellipses = []
                        for parcel in self.weather_system.parcels:
                            if hasattr(parcel, 'x') and hasattr(parcel, 'y'):
                                ellipse = (
                                    getattr(parcel, 'x', 0),
                                    getattr(parcel, 'y', 0),
                                    getattr(parcel, 'size', 100),
                                    getattr(parcel, 'size', 100),
                                    0,  # angle
                                    getattr(parcel, 'density', 0.5),
                                    getattr(parcel, 'type', 'cumulus')
                                )
                                cloud_ellipses.append(ellipse)
                    
                    # Extract positions
                    cloud_positions = []
                    for ellipse in cloud_ellipses:
                        if len(ellipse) >= 2:
                            x_m, y_m = ellipse[0], ellipse[1]
                            # Convert to km
                            domain_size = getattr(CFG, 'DOMAIN_SIZE_M', 10000)
                            area_size = getattr(CFG, 'AREA_SIZE_KM', 10)
                            x_km = x_m / domain_size * area_size
                            y_km = y_m / domain_size * area_size
                            cloud_positions.append((x_km, y_km))
                
                # Get weather metrics
                if hasattr(self.weather_system, 'current_cloud_cover_pct'):
                    cloud_coverage = self.weather_system.current_cloud_cover_pct()
                elif hasattr(self.weather_system, 'get_cloud_coverage'):
                    cloud_coverage = self.weather_system.get_cloud_coverage()
                
                if hasattr(self.weather_system, 'get_avg_trajectory'):
                    trajectory = self.weather_system.get_avg_trajectory()
                    if trajectory and len(trajectory) >= 2:
                        wind_speed, wind_direction = trajectory[0], trajectory[1]
                        
            except Exception as e:
                logger.warning(f"Error getting weather data: {e}")
        
        # Calculate solar position
        solar_elevation = 45.0  # Default fallback
        solar_azimuth = 180.0   # Default fallback
        
        if self.power_simulator and hasattr(self.power_simulator, 'calculate_solar_position'):
            try:
                solar_pos = self.power_simulator.calculate_solar_position(self.current_hour)
                solar_elevation = solar_pos.get('elevation', 45.0)
                solar_azimuth = solar_pos.get('azimuth', 180.0)
            except Exception as e:
                logger.warning(f"Error calculating solar position: {e}")
        
        # Calculate shadows
        panel_coverage = {}
        if self.shadow_calculator and cloud_ellipses and self.panel_df is not None:
            try:
                if hasattr(self.shadow_calculator, 'calculate_shadows'):
                    self.shadow_calculator.calculate_shadows()
                
                if hasattr(self.shadow_calculator, 'calculate_panel_coverage'):
                    # Convert ellipses for shadow calculation
                    ellipses_for_shadow = [e[:7] for e in cloud_ellipses if len(e) >= 7]
                    solar_position = {'elevation': solar_elevation, 'azimuth': solar_azimuth}
                    panel_coverage = self.shadow_calculator.calculate_panel_coverage(
                        ellipses_for_shadow, self.panel_df, solar_position
                    )
                elif hasattr(self.shadow_calculator, 'get_coverage'):
                    panel_coverage = self.shadow_calculator.get_coverage()
                        
            except Exception as e:
                logger.warning(f"Error calculating shadows: {e}")
        
        # Calculate power
        power_output = {}
        total_power = 0.0
        performance_ratio = 0.0
        
        if self.power_simulator:
            try:
                if hasattr(self.power_simulator, 'calculate_power'):
                    power_output = self.power_simulator.calculate_power(
                        panel_df=self.panel_df,
                        coverage=panel_coverage,
                        sim_time=self.current_hour
                    )
                elif hasattr(self.power_simulator, 'step'):
                    power_output = self.power_simulator.step(
                        self.current_hour, panel_coverage
                    )
                
                # Extract totals
                if isinstance(power_output, dict):
                    total_power = power_output.get('total', power_output.get('total_power', 0.0))
                    performance_ratio = power_output.get('performance_ratio', 0.0)
                    
            except Exception as e:
                logger.warning(f"Error calculating power: {e}")
        
        # Package results
        result = {
            # Cloud and weather data
            'cloud_ellipses': cloud_ellipses,
            'cloud_positions': cloud_positions,
            'cloud_coverage': cloud_coverage,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            
            # Shadow data
            'panel_coverage': panel_coverage,
            
            # Power data
            'power_output': power_output,
            'total_power': total_power,
            'performance_ratio': performance_ratio,
            
            # Environmental data
            'solar_elevation': solar_elevation,
            'solar_azimuth': solar_azimuth,
            'current_hour': self.current_hour,
            'ambient_temperature': 25.0,  # TODO: Add temperature simulation
            
            # Simulation metadata
            'frame_id': self.frame_count,
            'physics_dt': dt,
            'sim_time': sim_time
        }
        
        self._last_state = result
        return result
    
    def _create_fallback_state(self) -> Dict[str, Any]:
        """Create fallback state when simulation fails."""
        return {
            'cloud_ellipses': [],
            'cloud_positions': [],
            'cloud_coverage': 0.0,
            'wind_speed': 0.0,
            'wind_direction': 0.0,
            'panel_coverage': {},
            'power_output': {},
            'total_power': 0.0,
            'performance_ratio': 0.0,
            'solar_elevation': 45.0,
            'solar_azimuth': 180.0,
            'current_hour': self.current_hour,
            'ambient_temperature': 25.0,
            'frame_id': self.frame_count,
            'physics_dt': self.config.physics_timestep,
            'sim_time': self.frame_count * self.config.physics_timestep
        }
    
    def _update_performance_metrics(self, frame_time_ms: float, result: Dict[str, Any]):
        """Update performance monitoring metrics."""
        self.performance_metrics.update_frame_timing(frame_time_ms)
        
        # Update simulation metrics
        self.performance_metrics.cloud_count = len(result.get('cloud_ellipses', []))
        self.performance_metrics.active_panels = len(self.panel_df) if self.panel_df is not None else 0
        self.performance_metrics.total_power_kw = result.get('total_power', 0.0)
        self.performance_metrics.cloud_coverage_pct = result.get('cloud_coverage', 0.0)
        self.performance_metrics.wind_speed_ms = result.get('wind_speed', 0.0)
        
        # Update worker metrics if available
        if self.worker:
            try:
                worker_stats = self.worker.get_performance_stats()
                self.performance_metrics.worker_dt_queue_size = worker_stats.get('dt_queue_size', 0)
                self.performance_metrics.worker_result_queue_size = worker_stats.get('result_queue_size', 0)
                self.performance_metrics.worker_queue_pressure = worker_stats.get('dt_queue_pressure', 0.0)
                self.performance_metrics.worker_processing_time_ms = worker_stats.get('avg_processing_ms', 0.0)
                self.performance_metrics.worker_error_count = worker_stats.get('error_count', 0)
                self.performance_metrics.worker_recovery_count = worker_stats.get('recovery_count', 0)
            except Exception as e:
                if self.config.debug_mode:
                    logger.debug(f"Error getting worker stats: {e}")
        
        # Store performance history
        current_time = self._get_current_time()
        if (current_time - self.last_performance_update) >= self.config.performance_update_interval:
            self.performance_history.append({
                'time': current_time,
                'metrics': self.performance_metrics.to_dict()
            })
            self.last_performance_update = current_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_metrics.to_dict()
        stats.update({
            'simulation_time_hours': self.current_hour,
            'frame_count': self.frame_count,
            'threading_enabled': self.config.use_threading,
            'worker_active': self.worker is not None and self.worker.is_alive() if self.worker else False
        })
        return stats
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get detailed debug information."""
        debug_info = {
            'config': self.config.to_dict(),
            'performance': self.get_performance_stats(),
            'simulation_state': {
                'current_hour': self.current_hour,
                'frame_count': self.frame_count,
                'simulation_start_time': self.simulation_start_time,
                'last_update_time': self.last_update_time
            },
            'components': {
                'weather_system': type(self.weather_system).__name__ if self.weather_system else None,
                'shadow_calculator': type(self.shadow_calculator).__name__ if self.shadow_calculator else None,
                'power_simulator': type(self.power_simulator).__name__ if self.power_simulator else None,
                'panel_count': len(self.panel_df) if self.panel_df is not None else 0
            }
        }
        
        # Add worker debug info
        if self.worker:
            try:
                debug_info['worker'] = self.worker.get_debug_info()
            except Exception as e:
                debug_info['worker'] = {'error': str(e)}
        
        return debug_info
    
    def shutdown(self):
        """Gracefully shutdown the simulation controller."""
        logger.info("Shutting down SimulationController...")
        self._shutdown_requested = True
        
        # Stop physics worker
        if self.worker:
            try:
                self.worker.stop()
                logger.info("✓ Physics worker thread stopped")
            except Exception as e:
                logger.error(f"Error stopping physics worker: {e}")
        
        # Cleanup components
        try:
            if hasattr(self.weather_system, 'cleanup'):
                self.weather_system.cleanup()
            if hasattr(self.shadow_calculator, 'cleanup'):
                self.shadow_calculator.cleanup()
            if hasattr(self.power_simulator, 'cleanup'):
                self.power_simulator.cleanup()
        except Exception as e:
            logger.warning(f"Error during component cleanup: {e}")
        
        logger.info("✓ SimulationController shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Factory function
def create_simulation_controller(config_path: Optional[str] = None,
                                 panel_df: Optional[pd.DataFrame] = None,
                                 use_threading: bool = True,
                                 debug_mode: bool = False) -> SimulationController:
    """
    Factory function to create a properly configured simulation controller.
    
    Args:
        config_path: Path to configuration file
        panel_df: Panel DataFrame
        use_threading: Enable threading mode
        debug_mode: Enable debug mode
        
    Returns:
        Configured SimulationController instance
    """
    if config_path and os.path.exists(config_path):
        config = SimulationConfig.from_file(config_path)
    else:
        config = SimulationConfig()
    
    # Apply overrides
    config.use_threading = use_threading
    config.debug_mode = debug_mode
    
    return SimulationController(config=config, panel_df=panel_df)


# Example usage and testing
if __name__ == "__main__":
    # Test both threaded and non-threaded modes
    logger.info("Testing SimulationController...")
    
    # Create test panel data
    test_panels = pd.DataFrame({
        'panel_id': ['P001', 'P002', 'P003'],
        'x_km': [1.0, 1.5, 2.0],
        'y_km': [1.0, 1.5, 2.0],
        'x': [100, 150, 200],
        'y': [100, 150, 200],
        'power_capacity': [5.0, 5.0, 5.0]
    })
    
    # Test non-threaded mode
    logger.info("Testing non-threaded mode...")
    controller_direct = create_simulation_controller(
        panel_df=test_panels,
        use_threading=False,
        debug_mode=True
    )
    
    try:
        for i in range(5):
            result = controller_direct.step(1/60.0)
            logger.info(f"Step {i}: clouds={len(result['cloud_ellipses'])}, power={result['total_power']:.1f}kW")
        
        stats = controller_direct.get_performance_stats()
        logger.info(f"Performance: {stats['avg_frame_time_ms']:.2f}ms avg")
        
    finally:
        controller_direct.shutdown()
    
    # Test threaded mode
    logger.info("Testing threaded mode...")
    controller_threaded = create_simulation_controller(
        panel_df=test_panels,
        use_threading=True,
        debug_mode=True
    )
    
    try:
        for i in range(5):
            result = controller_threaded.step(1/60.0)
            logger.info(f"Step {i}: clouds={len(result['cloud_ellipses'])}, power={result['total_power']:.1f}kW")
        
        stats = controller_threaded.get_performance_stats()
        logger.info(f"Performance: {stats['avg_frame_time_ms']:.2f}ms avg")
        logger.info(f"Worker queue: dt={stats['worker_dt_queue_size']}, result={stats['worker_result_queue_size']}")
        
    finally:
        controller_threaded.shutdown()
    
    logger.info("✓ SimulationController test completed")