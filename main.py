#!/usr/bin/env python3
"""
Solar Farm Cloud Simulation - Complete Threading Implementation
Advanced game loop with comprehensive threading architecture support.

Features:
- Complete command-line argument parsing with threading control
- Advanced GameLoop class with FPS management and performance monitoring  
- Signal handling for graceful shutdown
- Configuration loading from JSON files
- Debug overlay with performance statistics
- Input handling with threading-aware features
- Fixed timestep simulation with background physics worker
- Comprehensive error handling and logging
- Windows Unicode compatibility fixes

Architecture:
┌─────────────────┐    dt     ┌─────────────────┐
│   Main Thread   │─────────→│  Physics Thread │
│ (Rendering & I/O)│          │   (Worker)      │
└─────────────────┘←─────────┐                 │
         │  result           │                 │
         └───────────────────┘─────────────────┘
"""
import argparse
import time
import traceback
import logging
import json
import os
import sys
import signal
import atexit
import math
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from dataclasses import dataclass

# Windows Unicode console fix (CRITICAL FIX)
if sys.platform == "win32":
    try:
        # Set console code page to UTF-8
        os.system("chcp 65001 > nul 2>&1")
        # Set Python I/O encoding to UTF-8
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # Force stdout encoding
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception as e:
        # Fallback for older Python versions
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
        except:
            pass  # If all else fails, continue without Unicode fix

# Third-party imports
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARN] Pygame not available - headless mode only")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[FAIL] NumPy not available - simulation will not work")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("[FAIL] Pandas not available - simulation will not work")
    sys.exit(1)

# Local imports
try:
    import sim_config as CFG
except ImportError:
    # Fallback configuration
    class CFG:
        FPS = 60
        AREA_SIZE_KM = 10
        DOMAIN_SIZE_M = 10000
        TIME_SCALE = 1.0
        PHYSICS_TIMESTEP = 1/60.0

# Import rendering and background modules
try:
    from cloud_renderer import draw_vectorized_clouds, draw_cloud_trail, km_to_screen_coords
    CLOUD_RENDERER_AVAILABLE = True
except ImportError:
    CLOUD_RENDERER_AVAILABLE = False
    # Fallback implementations
    def draw_vectorized_clouds(*args, **kwargs):
        pass
    def draw_cloud_trail(*args, **kwargs):
        pass
    def km_to_screen_coords(x_km, y_km, x_range, y_range, width, height):
        return int((x_km - x_range[0]) / (x_range[1] - x_range[0]) * width), \
               int((y_km - y_range[0]) / (y_range[1] - y_range[0]) * height)

try:
    from background_module import setup_background_fast, create_panel_dataframe
    BACKGROUND_MODULE_AVAILABLE = True
except ImportError:
    BACKGROUND_MODULE_AVAILABLE = False
    def setup_background_fast(*args, **kwargs):
        pass
    def create_panel_dataframe(*args, **kwargs):
        return pd.DataFrame()

# Configure logging with ASCII-safe formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simulation.log', encoding='utf-8', errors='replace')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GameConfiguration:
    """Complete game configuration with validation."""
    # Display settings
    width: int = 1200
    height: int = 900
    fullscreen: bool = False
    fps: int = 60
    vsync: bool = True
    
    # Threading settings
    use_threading: bool = True
    worker_stats_display: bool = False
    
    # Simulation settings
    start_hour: float = 11.0
    time_scale: float = 1.0
    single_cloud_mode: bool = False
    test_gliding: bool = False
    
    # Debug and profiling
    debug_mode: bool = False
    profile_mode: bool = False
    show_debug_overlay: bool = False
    headless_mode: bool = False
    
    # Data settings
    panel_file: Optional[str] = None
    config_file: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Testing settings
    test_duration: int = 0
    
    @classmethod
    def from_args(cls, args) -> 'GameConfiguration':
        """Create configuration from command line arguments."""
        # Parse resolution
        try:
            width, height = map(int, args.resolution.split('x'))
        except (ValueError, AttributeError):
            width, height = 1200, 900
        
        return cls(
            width=width,
            height=height,
            fullscreen=args.fullscreen,
            fps=args.fps,
            use_threading=not args.no_threading,
            worker_stats_display=args.worker_stats or args.debug,
            start_hour=args.start_hour,
            time_scale=args.time_scale,
            single_cloud_mode=args.single_cloud,
            test_gliding=args.test_gliding,
            debug_mode=args.debug,
            profile_mode=args.profile,
            show_debug_overlay=args.debug,
            headless_mode=args.no_gui,
            panel_file=args.panel_file,
            config_file=args.config,
            output_dir=args.output_dir,
            test_duration=args.test_duration
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'GameConfiguration':
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Convert to configuration object
            config = cls()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            logger.info(f"[OK] Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.warning(f"[WARN] Could not load config from {config_path}: {e}")
            return cls()  # Return default config
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        try:
            config_dict = {
                field.name: getattr(self, field.name)
                for field in self.__dataclass_fields__.values()
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"[OK] Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"[FAIL] Could not save config to {config_path}: {e}")
    
    def validate(self):
        """Validate configuration parameters."""
        # Validate display settings
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        
        # Validate simulation settings
        if not 0 <= self.start_hour <= 24:
            raise ValueError("Start hour must be between 0 and 24")
        
        if self.time_scale <= 0:
            raise ValueError("Time scale must be positive")
        
        logger.debug("[OK] Configuration validation passed")


class PerformanceDebugOverlay:
    """Advanced debug overlay with comprehensive performance metrics."""
    
    def __init__(self, font_size: int = 16):
        self.font_size = font_size
        self.font = None
        self.small_font = None
        self.visible = False
        self.stats_history = deque(maxlen=120)  # 2 seconds at 60fps
        
    def initialize(self):
        """Initialize fonts (call after pygame.init())."""
        if not PYGAME_AVAILABLE:
            return
        
        try:
            self.font = pygame.font.SysFont('Arial', self.font_size)
            self.small_font = pygame.font.SysFont('Arial', self.font_size - 4)
        except:
            # Fallback to default font
            try:
                self.font = pygame.font.Font(None, self.font_size + 4)
                self.small_font = pygame.font.Font(None, self.font_size)
            except:
                logger.warning("[WARN] Could not initialize debug overlay fonts")
    
    def toggle_visibility(self):
        """Toggle overlay visibility."""
        self.visible = not self.visible
        logger.info(f"[OK] Debug overlay: {'visible' if self.visible else 'hidden'}")
    
    def update_stats(self, performance_stats: Dict[str, Any]):
        """Update performance statistics."""
        self.stats_history.append({
            'timestamp': time.time(),
            'stats': performance_stats.copy()
        })
    
    def render(self, screen, performance_stats: Dict[str, Any]):
        """Render debug overlay with comprehensive performance info."""
        if not self.visible or not self.font or not PYGAME_AVAILABLE:
            return
        
        try:
            # Semi-transparent background
            overlay_width = 400
            overlay_height = 500
            overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
            overlay_surface.fill((0, 0, 0, 180))
            
            y_offset = 10
            line_height = self.font_size + 2
            
            def draw_text(text: str, color: Tuple[int, int, int] = (255, 255, 255)):
                nonlocal y_offset
                if y_offset < overlay_height - line_height:
                    text_surface = self.font.render(str(text), True, color)
                    overlay_surface.blit(text_surface, (10, y_offset))
                    y_offset += line_height
            
            def draw_small_text(text: str, color: Tuple[int, int, int] = (200, 200, 200)):
                nonlocal y_offset
                if y_offset < overlay_height - line_height and self.small_font:
                    text_surface = self.small_font.render(str(text), True, color)
                    overlay_surface.blit(text_surface, (15, y_offset))
                    y_offset += line_height - 2
            
            # Header
            draw_text("SOLAR FARM SIMULATION DEBUG", (255, 255, 0))
            y_offset += 5
            
            # FPS and Frame Timing
            draw_text("FRAME PERFORMANCE:", (0, 255, 255))
            current_fps = performance_stats.get('current_fps', 0)
            average_fps = performance_stats.get('average_fps', 0)
            min_fps = performance_stats.get('min_fps', 0)
            max_fps = performance_stats.get('max_fps', 0)
            
            fps_color = (255, 255, 255)
            if current_fps < 30:
                fps_color = (255, 100, 100)  # Red for poor performance
            elif current_fps < 50:
                fps_color = (255, 255, 100)  # Yellow for moderate performance
            else:
                fps_color = (100, 255, 100)  # Green for good performance
            
            draw_small_text(f"Current FPS: {current_fps:.1f}", fps_color)
            draw_small_text(f"Average FPS: {average_fps:.1f}")
            draw_small_text(f"Min/Max FPS: {min_fps:.1f}/{max_fps:.1f}")
            y_offset += 5
            
            # Threading Information
            threading_enabled = performance_stats.get('threading_enabled', False)
            draw_text("THREADING STATUS:", (0, 255, 255))
            
            if threading_enabled:
                draw_small_text("Threading: ENABLED", (100, 255, 100))
                
                # Worker thread stats
                worker_active = performance_stats.get('worker_active', False)
                worker_fps = performance_stats.get('worker_fps', 0)
                worker_errors = performance_stats.get('worker_errors', 0)
                
                worker_color = (100, 255, 100) if worker_active else (255, 100, 100)
                draw_small_text(f"Worker: {'ACTIVE' if worker_active else 'INACTIVE'}", worker_color)
                draw_small_text(f"Worker FPS: {worker_fps:.1f}")
                
                if worker_errors > 0:
                    draw_small_text(f"Worker Errors: {worker_errors}", (255, 100, 100))
                
                # Queue information
                dt_queue_size = performance_stats.get('dt_queue_size', 0)
                result_queue_size = performance_stats.get('result_queue_size', 0)
                dt_queue_pressure = performance_stats.get('dt_queue_pressure', 0) * 100
                result_queue_pressure = performance_stats.get('result_queue_pressure', 0) * 100
                queue_lag = performance_stats.get('queue_lag_ms', 0)
                
                draw_small_text(f"DT Queue: {dt_queue_size} ({dt_queue_pressure:.1f}%)")
                draw_small_text(f"Result Queue: {result_queue_size} ({result_queue_pressure:.1f}%)")
                if queue_lag > 0:
                    lag_color = (255, 100, 100) if queue_lag > 16 else (255, 255, 100)
                    draw_small_text(f"Queue Lag: {queue_lag:.1f}ms", lag_color)
            else:
                draw_small_text("Threading: DISABLED", (255, 255, 100))
            
            y_offset += 5
            
            # Physics Performance
            draw_text("PHYSICS PERFORMANCE:", (0, 255, 255))
            physics_time = performance_stats.get('physics_time_ms', 0)
            avg_physics_time = performance_stats.get('avg_physics_time_ms', 0)
            max_physics_time = performance_stats.get('max_physics_time_ms', 0)
            
            physics_color = (255, 255, 255)
            if physics_time > 16:  # > 60 FPS target
                physics_color = (255, 100, 100)
            elif physics_time > 10:
                physics_color = (255, 255, 100)
            else:
                physics_color = (100, 255, 100)
            
            draw_small_text(f"Physics Time: {physics_time:.1f}ms", physics_color)
            draw_small_text(f"Avg Physics: {avg_physics_time:.1f}ms")
            draw_small_text(f"Max Physics: {max_physics_time:.1f}ms")
            y_offset += 5
            
            # Simulation Information
            draw_text("SIMULATION STATUS:", (0, 255, 255))
            current_hour = performance_stats.get('current_hour', 0)
            sim_uptime = performance_stats.get('simulation_uptime', 0)
            total_frames = performance_stats.get('total_frames', 0)
            panels_count = performance_stats.get('panels_count', 0)
            total_energy = performance_stats.get('total_energy_kwh', 0)
            
            draw_small_text(f"Current Time: {current_hour:.1f}h")
            draw_small_text(f"Uptime: {sim_uptime:.1f}s")
            draw_small_text(f"Total Frames: {total_frames}")
            draw_small_text(f"Panels: {panels_count}")
            if total_energy > 0:
                draw_small_text(f"Energy: {total_energy:.2f} kWh")
            y_offset += 5
            
            # System Information
            draw_text("SYSTEM STATUS:", (0, 255, 255))
            memory_usage = performance_stats.get('memory_usage_mb', 0)
            cpu_usage = performance_stats.get('cpu_usage_pct', 0)
            
            if memory_usage > 0:
                draw_small_text(f"Memory: {memory_usage:.1f} MB")
            if cpu_usage > 0:
                draw_small_text(f"CPU: {cpu_usage:.1f}%")
            
            # Controls information
            y_offset += 10
            draw_text("CONTROLS:", (255, 255, 0))
            draw_small_text("F1: Toggle this overlay")
            draw_small_text("F2: Reset worker thread")
            draw_small_text("F3: Save screenshot")
            draw_small_text("SPACE: Pause simulation")
            draw_small_text("ESC: Quit")
            draw_small_text("UP/DOWN: Wind speed")
            draw_small_text("LEFT/RIGHT: Movement speed")
            draw_small_text("R: Reset simulation")
            
            # Blit overlay to screen
            screen.blit(overlay_surface, (10, 10))
            
        except Exception as e:
            logger.error(f"[FAIL] Error rendering debug overlay: {e}")


class GameLoop:
    """
    Advanced game loop with comprehensive threading support.
    
    Features:
    - Target FPS management and frame rate limiting
    - Fixed timestep simulation updates
    - Input handling and event processing
    - Rendering coordination with threading awareness
    - Performance monitoring and debug overlay
    - Signal handling for graceful shutdown
    """
    
    def __init__(self, config: GameConfiguration):
        """Initialize game loop with configuration."""
        self.config = config
        self.running = True
        self.paused = False
        
        # Initialize pygame if not headless and available
        if not config.headless_mode and PYGAME_AVAILABLE:
            try:
                pygame.init()
                
                # Set up display
                display_flags = pygame.RESIZABLE
                if config.fullscreen:
                    display_flags |= pygame.FULLSCREEN
                
                self.screen = pygame.display.set_mode(
                    (config.width, config.height), display_flags
                )
                pygame.display.set_caption("Solar Farm Cloud Simulation - Threading Enhanced")
                
                self.clock = pygame.time.Clock()
                logger.info(f"[OK] Pygame initialized: {config.width}x{config.height}")
                
            except Exception as e:
                logger.error(f"[FAIL] Could not initialize pygame: {e}")
                self.screen = None
                self.clock = None
                config.headless_mode = True
        else:
            self.screen = None
            self.clock = None
            if not config.headless_mode:
                logger.warning("[WARN] Pygame not available, forcing headless mode")
                config.headless_mode = True
        
        # Initialize controller and components
        self.controller = None
        self.panel_df = None
        
        # Performance tracking
        self.frame_count = 0
        self.frame_times = deque(maxlen=120)  # 2 seconds at 60fps
        self.simulation_start_time = time.time()
        self.last_performance_update = time.time()
        
        # Debug overlay
        self.debug_overlay = PerformanceDebugOverlay()
        if not config.headless_mode:
            self.debug_overlay.initialize()
            self.debug_overlay.visible = config.show_debug_overlay
        
        # Simulation bounds
        self.x_range = (0, getattr(CFG, 'AREA_SIZE_KM', 10))
        self.y_range = (0, getattr(CFG, 'AREA_SIZE_KM', 10))
        
        # Colors and constants
        self.SKY_COLOR = (230, 242, 255)
        self.GRID_COLOR = (200, 200, 200)
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup_on_exit)
        
        logger.info(f"[OK] GameLoop initialized:")
        logger.info(f"  Resolution: {config.width}x{config.height}")
        logger.info(f"  Threading: {'enabled' if config.use_threading else 'disabled'}")
        logger.info(f"  Debug mode: {'enabled' if config.debug_mode else 'disabled'}")
        logger.info(f"  Headless: {'enabled' if config.headless_mode else 'disabled'}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"[WARN] Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def _cleanup_on_exit(self):
        """Ensure proper cleanup on exit."""
        try:
            if self.controller and hasattr(self.controller, 'shutdown'):
                self.controller.shutdown()
            
            if not self.config.headless_mode and PYGAME_AVAILABLE:
                pygame.quit()
                
            logger.info("[OK] Cleanup completed")
        except Exception as e:
            logger.error(f"[FAIL] Error during cleanup: {e}")
    
    def initialize_simulation(self):
        """Initialize simulation controller and load data."""
        logger.info("[OK] Initializing simulation components...")
        
        try:
            # Load panel data
            self.panel_df = self._load_panel_data()
            
            # Initialize simulation controller
            self._initialize_controller()
            
            # Apply configuration settings
            self._apply_config_settings()
            
            logger.info("[OK] Simulation initialization complete")
            
        except Exception as e:
            logger.error(f"[FAIL] Simulation initialization failed: {e}")
            raise
    
    def _load_panel_data(self) -> pd.DataFrame:
        """Load panel data from file or create default data."""
        panel_file = self.config.panel_file or "extended_coordinates.csv"
        
        if os.path.exists(panel_file):
            try:
                panel_df = pd.read_csv(panel_file)
                logger.info(f"[OK] Loaded {len(panel_df)} panels from {panel_file}")
                
                # Validate required columns
                required_cols = ['panel_id', 'x_km', 'y_km', 'x', 'y']
                missing_cols = [col for col in required_cols if col not in panel_df.columns]
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")
                
                # Handle missing power_capacity column gracefully (CRITICAL FIX)
                if 'power_capacity' not in panel_df.columns:
                    panel_df = panel_df.copy()
                    panel_df['power_capacity'] = 5.0  # Default 5.0 kW as specified
                    logger.info("[OK] Added default power_capacity (5.0 kW) for all panels")
                
                return panel_df
                
            except Exception as e:
                logger.error(f"[FAIL] Error loading {panel_file}: {e}")
                logger.info("[WARN] Falling back to generated panel data")
        else:
            logger.info(f"[WARN] Panel file {panel_file} not found, creating default data")
        
        # Create default panel data
        return self._create_default_panel_data()
    
    def _create_default_panel_data(self, num_panels: int = 36) -> pd.DataFrame:
        """Create default panel data for testing."""
        grid_size = int(math.sqrt(num_panels))
        
        panel_data = []
        for i in range(grid_size):
            for j in range(grid_size):
                if i * grid_size + j < num_panels:
                    panel_data.append({
                        'panel_id': f"P{i*grid_size+j+1:03d}",
                        'x_km': 2.0 + i * 1.0,
                        'y_km': 2.0 + j * 1.0,
                        'x': 2000 + i * 1000,
                        'y': 2000 + j * 1000,
                        'power_capacity': 5.0  # Default 5.0 kW as specified
                    })
        
        logger.info(f"[OK] Generated {len(panel_data)} default panels")
        return pd.DataFrame(panel_data)
    
    def _initialize_controller(self):
        """Initialize simulation controller with threading support."""
        try:
            from simulation_controller import create_simulation_controller
            
            # Create controller configuration
            controller_config = {
                'use_threading': self.config.use_threading,
                'start_time': self.config.start_hour,
                'debug_mode': self.config.debug_mode,
                'time_scale': self.config.time_scale
            }
            
            # Load from config file if specified
            if self.config.config_file and os.path.exists(self.config.config_file):
                try:
                    with open(self.config.config_file, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                    controller_config.update(file_config)
                    logger.info(f"[OK] Loaded controller config from {self.config.config_file}")
                except Exception as e:
                    logger.warning(f"[WARN] Could not load controller config: {e}")
            
            self.controller = create_simulation_controller(
                panel_df=self.panel_df,
                use_threading=self.config.use_threading,
                debug_mode=self.config.debug_mode
            )
            
            logger.info("[OK] Simulation controller initialized")
            
        except ImportError as e:
            logger.error(f"[FAIL] Could not import simulation_controller: {e}")
            raise
        except Exception as e:
            logger.error(f"[FAIL] Controller initialization failed: {e}")
            raise
    
    def _apply_config_settings(self):
        """Apply configuration settings to simulation."""
        try:
            # Set global configuration
            CFG.TIME_SCALE = self.config.time_scale
            CFG.FPS = self.config.fps
            
            if self.config.single_cloud_mode:
                CFG.SINGLE_CLOUD_MODE = True
                logger.info("[OK] Single cloud mode enabled")
            
            if self.config.test_gliding:
                CFG.SINGLE_CLOUD_MODE = True
                if hasattr(CFG, 'SPAWN_PROBABILITY'):
                    CFG.SPAWN_PROBABILITY = 0.8
                logger.info("[OK] Test gliding mode enabled")
                
        except Exception as e:
            logger.warning(f"[WARN] Error applying config settings: {e}")
    
    def handle_events(self):
        """Handle input events with threading-aware features."""
        if self.config.headless_mode or not self.screen or not PYGAME_AVAILABLE:
            return
        
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
                elif event.type == pygame.KEYDOWN:
                    self._handle_key_press(event.key)
                    
                elif event.type == pygame.VIDEORESIZE:
                    self.config.width, self.config.height = event.size
                    self.screen = pygame.display.set_mode(
                        (self.config.width, self.config.height), pygame.RESIZABLE
                    )
                    logger.info(f"[OK] Window resized to {self.config.width}x{self.config.height}")
                    
        except Exception as e:
            logger.error(f"[FAIL] Error handling events: {e}")
    
    def _handle_key_press(self, key: int):
        """Handle individual key press events."""
        try:
            # Basic controls
            if key == pygame.K_ESCAPE:
                self.running = False
                logger.info("[OK] Quit requested")
            
            elif key == pygame.K_SPACE:
                self.paused = not self.paused
                logger.info(f"[OK] Simulation {'paused' if self.paused else 'resumed'}")
            
            # Debug controls
            elif key == pygame.K_F1:
                self.debug_overlay.toggle_visibility()
            
            elif key == pygame.K_F2 and self.config.use_threading:
                if hasattr(self.controller, 'reset_worker'):
                    logger.info("[WARN] Force resetting physics worker...")
                    self.controller.reset_worker()
                else:
                    logger.warning("[WARN] Worker reset not available")
            
            elif key == pygame.K_F3:
                self._save_screenshot()
            
            # Simulation parameter controls
            elif key == pygame.K_UP:
                self._adjust_wind_speed(0.5)
            elif key == pygame.K_DOWN:
                self._adjust_wind_speed(-0.5)
            elif key == pygame.K_RIGHT:
                self._adjust_movement(0.5)
            elif key == pygame.K_LEFT:
                self._adjust_movement(-0.5)
            
            # Reset simulation
            elif key == pygame.K_r:
                if hasattr(self.controller, 'reset_simulation'):
                    self.controller.reset_simulation()
                    logger.info("[OK] Simulation reset")
                else:
                    logger.info("[WARN] Simulation reset not available")
                    
        except Exception as e:
            logger.error(f"[FAIL] Error handling key press: {e}")
    
    def _adjust_wind_speed(self, delta: float):
        """Adjust wind speed within limits."""
        try:
            if hasattr(CFG, 'BASE_WIND_SPEED'):
                old_speed = CFG.BASE_WIND_SPEED
                CFG.BASE_WIND_SPEED = max(0.5, min(10.0, CFG.BASE_WIND_SPEED + delta))
                logger.info(f"[OK] Wind speed: {CFG.BASE_WIND_SPEED:.1f} m/s (was {old_speed:.1f})")
            else:
                logger.warning("[WARN] Wind speed adjustment not available")
        except Exception as e:
            logger.error(f"[FAIL] Error adjusting wind speed: {e}")
    
    def _adjust_movement(self, delta: float):
        """Adjust movement multiplier within limits."""
        try:
            if hasattr(CFG, 'MOVEMENT_MULTIPLIER'):
                old_mult = CFG.MOVEMENT_MULTIPLIER
                CFG.MOVEMENT_MULTIPLIER = max(0.5, min(3.0, CFG.MOVEMENT_MULTIPLIER + delta))
                logger.info(f"[OK] Movement multiplier: {CFG.MOVEMENT_MULTIPLIER:.1f} (was {old_mult:.1f})")
            else:
                logger.warning("[WARN] Movement adjustment not available")
        except Exception as e:
            logger.error(f"[FAIL] Error adjusting movement: {e}")
    
    def _save_screenshot(self):
        """Save screenshot to output directory."""
        if self.config.headless_mode or not self.screen or not PYGAME_AVAILABLE:
            logger.warning("[WARN] Screenshot not available in headless mode")
            return
        
        try:
            output_dir = self.config.output_dir or "screenshots"
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"solar_farm_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            pygame.image.save(self.screen, filepath)
            logger.info(f"[OK] Screenshot saved: {filepath}")
            
        except Exception as e:
            logger.error(f"[FAIL] Could not save screenshot: {e}")
    
    def update_simulation(self, dt: float) -> Dict[str, Any]:
        """Update simulation with fixed timestep."""
        if self.paused:
            # Return last known state when paused
            return getattr(self, '_last_simulation_result', {})
        
        try:
            # Submit timestep to controller (handles threading internally)
            result = self.controller.step(dt)
            self._last_simulation_result = result
            return result
            
        except Exception as e:
            logger.error(f"[FAIL] Error updating simulation: {e}")
            if self.config.debug_mode:
                traceback.print_exc()
            
            # Return safe fallback state
            return {
                'cloud_ellipses': [],
                'cloud_positions': [],
                'panel_coverage': {},
                'power_output': {},
                'total_power': 0.0,
                'current_hour': 12.0,
                'error': True,
                'error_message': str(e)
            }
    
    def render_frame(self, simulation_result: Dict[str, Any]):
        """Render complete simulation frame."""
        if self.config.headless_mode or not self.screen or not PYGAME_AVAILABLE:
            return
        
        try:
            # Clear screen
            self.screen.fill(self.SKY_COLOR)
            
            # Draw background grid
            if BACKGROUND_MODULE_AVAILABLE:
                setup_background_fast(
                    self.screen, self.config.width, self.config.height,
                    self.x_range, self.y_range, self.GRID_COLOR
                )
            
            # Draw solar panels
            self._render_solar_panels(simulation_result)
            
            # Draw clouds
            if CLOUD_RENDERER_AVAILABLE:
                self._render_clouds(simulation_result)
            
            # Draw UI elements
            self._render_ui_elements(simulation_result)
            
            # Draw debug overlay
            if self.debug_overlay.visible:
                performance_stats = self._get_performance_stats(simulation_result)
                self.debug_overlay.render(self.screen, performance_stats)
            
        except Exception as e:
            logger.error(f"[FAIL] Error rendering frame: {e}")
            if self.config.debug_mode:
                traceback.print_exc()
    
    def _get_performance_stats(self, simulation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance statistics for debug overlay."""
        try:
            # Get base stats from controller
            if hasattr(self.controller, 'get_performance_stats'):
                stats = self.controller.get_performance_stats()
            else:
                stats = {}
            
            # Add frame-level stats
            if self.frame_times:
                current_fps = 1.0 / (self.frame_times[-1] + 1e-6)
                average_fps = len(self.frame_times) / (sum(self.frame_times) + 1e-6)
                min_fps = 1.0 / (max(self.frame_times) + 1e-6)
                max_fps = 1.0 / (min(self.frame_times) + 1e-6)
            else:
                current_fps = average_fps = min_fps = max_fps = 0.0
            
            stats.update({
                'current_fps': current_fps,
                'average_fps': average_fps,
                'min_fps': min_fps,
                'max_fps': max_fps,
                'threading_enabled': self.config.use_threading,
                'total_frames': self.frame_count,
                'simulation_uptime': time.time() - self.simulation_start_time,
                'current_hour': simulation_result.get('current_hour', 0),
                'panels_count': len(self.panel_df) if self.panel_df is not None else 0
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"[FAIL] Error getting performance stats: {e}")
            return {}
    
    def _render_solar_panels(self, simulation_result: Dict[str, Any]):
        """Render solar panels with power visualization."""
        if self.panel_df is None or len(self.panel_df) == 0 or not PYGAME_AVAILABLE:
            return
        
        try:
            panel_coverage = simulation_result.get('panel_coverage', {})
            power_output = simulation_result.get('power_output', {})
            
            for _, panel in self.panel_df.iterrows():
                panel_id = panel['panel_id']
                x_km, y_km = panel['x_km'], panel['y_km']
                
                # Convert to screen coordinates
                x, y = km_to_screen_coords(
                    x_km, y_km, self.x_range, self.y_range,
                    self.config.width, self.config.height
                )
                
                # Calculate panel color based on power output
                color = self._calculate_panel_color(panel_id, panel_coverage, power_output)
                
                # Draw panel rectangle
                panel_size = 8
                pygame.draw.rect(
                    self.screen, color,
                    (x - panel_size//2, y - panel_size//2, panel_size, panel_size)
                )
                
        except Exception as e:
            logger.debug(f"[WARN] Error rendering panels: {e}")
    
    def _calculate_panel_color(self, panel_id: str, panel_coverage: Dict, power_output: Dict) -> Tuple[int, int, int]:
        """Calculate panel color based on power output and cloud coverage."""
        try:
            # Get coverage and power values
            coverage = panel_coverage.get(panel_id, 0.0)
            
            if isinstance(power_output, dict) and panel_id in power_output:
                panel_power = power_output[panel_id]
                if isinstance(panel_power, dict):
                    power = panel_power.get('final_power', 0.0)
                    baseline = panel_power.get('baseline', 1.0)
                    power_ratio = power / baseline if baseline > 0 else 0.0
                else:
                    power_ratio = float(panel_power) / 5.0 if panel_power > 0 else 0.0  # Assume 5kW baseline
            else:
                power_ratio = max(0.0, 1.0 - coverage)  # Estimate from coverage
            
            # Clamp ratio
            power_ratio = max(0.0, min(1.0, power_ratio))
            
            # Color interpolation: red (no power) -> yellow (partial) -> green (full)
            if power_ratio < 0.5:
                # Red to yellow
                ratio = power_ratio * 2
                return (255, int(255 * ratio), 0)
            else:
                # Yellow to green
                ratio = (power_ratio - 0.5) * 2
                return (int(255 * (1 - ratio)), 255, 0)
                
        except Exception as e:
            logger.debug(f"[WARN] Error calculating panel color: {e}")
            return (128, 128, 128)  # Gray fallback
    
    def _render_clouds(self, simulation_result: Dict[str, Any]):
        """Render clouds and cloud trails."""
        try:
            cloud_ellipses = simulation_result.get('cloud_ellipses', [])
            cloud_positions = simulation_result.get('cloud_positions', [])
            
            if cloud_ellipses:
                draw_vectorized_clouds(
                    self.screen, cloud_ellipses,
                    self.x_range, self.y_range,
                    self.config.width, self.config.height
                )
            
            if cloud_positions:
                draw_cloud_trail(
                    self.screen, cloud_positions,
                    self.x_range, self.y_range,
                    self.config.width, self.config.height
                )
                
        except Exception as e:
            logger.debug(f"[WARN] Error rendering clouds: {e}")
    
    def _render_ui_elements(self, simulation_result: Dict[str, Any]):
        """Render UI elements like status text."""
        if not PYGAME_AVAILABLE:
            return
        
        try:
            if not hasattr(self, '_ui_font'):
                try:
                    self._ui_font = pygame.font.SysFont('Arial', 20)
                    self._small_font = pygame.font.SysFont('Arial', 16)
                except:
                    self._ui_font = pygame.font.Font(None, 24)
                    self._small_font = pygame.font.Font(None, 20)
            
            # Current time display
            current_hour = simulation_result.get('current_hour', 0)
            time_text = f"Time: {current_hour:.1f}h"
            time_surface = self._ui_font.render(time_text, True, (255, 255, 255))
            
            # Add black outline for better visibility
            outline_surface = self._ui_font.render(time_text, True, (0, 0, 0))
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                self.screen.blit(outline_surface, (self.config.width - 120 + dx, 10 + dy))
            self.screen.blit(time_surface, (self.config.width - 120, 10))
            
            # Power output display
            total_power = simulation_result.get('total_power', 0)
            if total_power > 0:
                power_text = f"Power: {total_power:.1f} kW"
                power_surface = self._small_font.render(power_text, True, (255, 255, 255))
                outline_surface = self._small_font.render(power_text, True, (0, 0, 0))
                
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    self.screen.blit(outline_surface, (self.config.width - 140 + dx, 35 + dy))
                self.screen.blit(power_surface, (self.config.width - 140, 35))
            
            # Error display
            if simulation_result.get('error', False):
                error_msg = simulation_result.get('error_message', 'Unknown error')
                error_text = f"ERROR: {error_msg[:50]}..."  # Truncate long messages
                error_surface = self._small_font.render(error_text, True, (255, 100, 100))
                self.screen.blit(error_surface, (10, self.config.height - 30))
                
        except Exception as e:
            logger.debug(f"[WARN] Error rendering UI: {e}")
    
    def _update_performance_metrics(self, frame_time: float, simulation_result: Dict[str, Any]):
        """Update performance tracking metrics."""
        try:
            self.frame_times.append(frame_time)
            self.frame_count += 1
            
            # Update debug overlay stats
            if hasattr(self.controller, 'get_performance_stats'):
                performance_stats = self.controller.get_performance_stats()
                self.debug_overlay.update_stats(performance_stats)
                
        except Exception as e:
            logger.debug(f"[WARN] Error updating performance metrics: {e}")
    
    def run_headless_test(self, duration_seconds: int) -> bool:
        """Run headless test for specified duration."""
        logger.info(f"[OK] Running headless test for {duration_seconds} seconds...")
        
        start_time = time.time()
        test_frame_count = 0
        errors = []
        
        try:
            while time.time() - start_time < duration_seconds and self.running:
                frame_start = time.time()
                
                # Fixed timestep
                dt = 1.0 / self.config.fps
                
                try:
                    # Update simulation
                    result = self.update_simulation(dt)
                    
                    # Check for errors
                    if result.get('error', False):
                        errors.append(result.get('error_message', 'Unknown error'))
                        
                except Exception as e:
                    errors.append(str(e))
                    if self.config.debug_mode:
                        traceback.print_exc()
                
                test_frame_count += 1
                
                # Throttle to approximate target FPS
                frame_time = time.time() - frame_start
                target_frame_time = 1.0 / self.config.fps
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
                
                # Progress report every 100 frames
                if test_frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    progress = elapsed / duration_seconds * 100
                    logger.info(f"[OK] Test progress: {progress:.1f}% ({test_frame_count} frames)")
        
        except KeyboardInterrupt:
            logger.info("[WARN] Headless test interrupted by user")
        
        # Test results
        actual_duration = time.time() - start_time
        avg_fps = test_frame_count / actual_duration if actual_duration > 0 else 0
        
        logger.info(f"[OK] Headless test completed:")
        logger.info(f"  Duration: {actual_duration:.1f}s")
        logger.info(f"  Frames: {test_frame_count}")
        logger.info(f"  Avg FPS: {avg_fps:.1f}")
        logger.info(f"  Errors: {len(errors)}")
        
        if errors:
            logger.error("[FAIL] Errors encountered during test:")
            for i, error in enumerate(errors[:5]):  # Show first 5 errors
                logger.error(f"  {i+1}: {error}")
            if len(errors) > 5:
                logger.error(f"  ... and {len(errors) - 5} more")
            return False
        else:
            logger.info("[OK] Test completed successfully with no errors")
            return True
    
    def run(self) -> bool:
        """
        Main game loop with comprehensive threading support.
        
        Returns:
            True if simulation completed successfully, False otherwise
        """
        logger.info("[OK] Starting solar farm simulation...")
        
        try:
            # Initialize simulation
            self.initialize_simulation()
            
            # Handle headless test mode
            if self.config.test_duration > 0:
                return self.run_headless_test(self.config.test_duration)
            
            # Main game loop
            logger.info("[OK] Entering main game loop...")
            while self.running:
                frame_start = time.time()
                
                # Handle input events
                self.handle_events()
                
                if not self.running:
                    break
                
                # Calculate frame delta time
                if self.clock and not self.config.headless_mode:
                    dt_ms = self.clock.tick(self.config.fps)
                    dt = dt_ms / 1000.0
                else:
                    # Headless mode - fixed timestep
                    dt = 1.0 / self.config.fps
                    time.sleep(dt)
                
                # Cap dt to prevent large jumps
                dt = min(dt, 0.1)
                
                # Update simulation
                simulation_result = self.update_simulation(dt)
                
                # Render frame (skip in headless mode)
                if not self.config.headless_mode and self.screen:
                    self.render_frame(simulation_result)
                    pygame.display.flip()
                
                # Update performance metrics
                frame_time = time.time() - frame_start
                self._update_performance_metrics(frame_time, simulation_result)
                
                # Progress report in headless mode
                if self.config.headless_mode and self.frame_count % 300 == 0:  # Every 5 seconds at 60fps
                    uptime = time.time() - self.simulation_start_time
                    logger.info(f"[OK] Simulation running: {uptime:.0f}s, {self.frame_count} frames")
            
            logger.info("[OK] Simulation completed successfully")
            return True
            
        except KeyboardInterrupt:
            logger.info("[WARN] Simulation interrupted by user")
            return True
            
        except Exception as e:
            logger.error(f"[FAIL] Critical simulation error: {e}")
            if self.config.debug_mode:
                traceback.print_exc()
            return False
            
        finally:
            # Comprehensive cleanup
            logger.info("[OK] Shutting down simulation...")
            self._cleanup_on_exit()


def parse_command_line_arguments() -> argparse.Namespace:
    """
    Complete command-line argument parsing with threading support.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Solar Farm Cloud Simulation with Advanced Threading Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard threaded mode with 60 FPS
  python main.py --threading --fps 60

  # Single-threaded comparison mode
  python main.py --no-threading --fps 30

  # Debug mode with performance overlay
  python main.py --threading --debug --profile

  # Custom configuration
  python main.py --threading --config solar_farm_config.json --fullscreen

  # Performance testing
  python main.py --threading --fps 120 --debug --resolution 2560x1440
  
  # Headless testing
  python main.py --no-gui --test-duration 30
        """
    )
    
    # Threading options (mutually exclusive)
    threading_group = parser.add_mutually_exclusive_group()
    threading_group.add_argument(
        '--threading', dest='no_threading', action='store_false',
        help='Enable background physics thread (default)'
    )
    threading_group.add_argument(
        '--no-threading', action='store_true',
        help='Disable background physics thread (for debugging)'
    )
    
    # Display options
    parser.add_argument(
        '--fps', type=int, default=getattr(CFG, 'FPS', 60),
        help=f'Target FPS (default: {getattr(CFG, "FPS", 60)})'
    )
    parser.add_argument(
        '--resolution', type=str, default='1200x900',
        help='Window resolution WIDTHxHEIGHT (default: 1200x900)'
    )
    parser.add_argument(
        '--fullscreen', action='store_true',
        help='Run in fullscreen mode'
    )
    
    # Configuration options
    parser.add_argument(
        '--config', type=str,
        help='JSON configuration file'
    )
    
    # Debug and profiling options
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug overlay and verbose logging'
    )
    parser.add_argument(
        '--profile', action='store_true',
        help='Enable performance profiling'
    )
    parser.add_argument(
        '--worker-stats', action='store_true',
        help='Show detailed worker thread statistics'
    )
    
    # Simulation options
    parser.add_argument(
        '--start-hour', type=float, default=11.0,
        help='Starting hour of simulation (0-24, default: 11.0)'
    )
    parser.add_argument(
        '--time-scale', type=float, default=1.0,
        help='Time scale multiplier (default: 1.0)'
    )
    parser.add_argument(
        '--single-cloud', action='store_true',
        help='Single cloud mode for testing'
    )
    parser.add_argument(
        '--test-gliding', action='store_true',
        help='Test cloud movement patterns'
    )
    
    # Data options
    parser.add_argument(
        '--panel-file', type=str,
        help='Custom panel layout CSV file (default: extended_coordinates.csv)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        help='Directory for output files and screenshots'
    )
    
    # Testing options
    parser.add_argument(
        '--no-gui', action='store_true',
        help='Run headless simulation for testing'
    )
    parser.add_argument(
        '--test-duration', type=int, default=0,
        help='Run headless test for N seconds then exit'
    )
    
    return parser.parse_args()


def validate_system_requirements() -> bool:
    """Validate system requirements for threading."""
    try:
        import threading
        import queue
        
        # Test basic threading functionality
        test_queue = queue.Queue()
        test_event = threading.Event()
        
        def test_worker():
            test_queue.put("test")
            test_event.set()
        
        test_thread = threading.Thread(target=test_worker)
        test_thread.start()
        test_thread.join(timeout=1.0)
        
        if not test_event.is_set():
            logger.warning("[WARN] Threading test failed - consider using --no-threading")
            return False
        
        logger.info("[OK] Threading validation passed")
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] Threading validation failed: {e}")
        return False


def check_dependencies():
    """Check for required and optional dependencies."""
    logger.info("[OK] Checking dependencies...")
    
    # Required dependencies
    required_ok = True
    if not NUMPY_AVAILABLE:
        logger.error("[FAIL] NumPy is required but not available")
        required_ok = False
    if not PANDAS_AVAILABLE:
        logger.error("[FAIL] Pandas is required but not available")
        required_ok = False
    
    # Optional dependencies
    if not PYGAME_AVAILABLE:
        logger.warning("[WARN] Pygame not available - headless mode only")
    else:
        logger.info("[OK] Pygame available")
    
    if not CLOUD_RENDERER_AVAILABLE:
        logger.warning("[WARN] Cloud renderer module not available")
    else:
        logger.info("[OK] Cloud renderer available")
    
    if not BACKGROUND_MODULE_AVAILABLE:
        logger.warning("[WARN] Background module not available")
    else:
        logger.info("[OK] Background module available")
    
    return required_ok


def main():
    """Main entry point for solar farm simulation."""
    print("="*60)
    print("SOLAR FARM CLOUD SIMULATION - THREADING ENHANCED")
    print("Advanced physics worker thread implementation")
    print("="*60)
    
    try:
        # Check dependencies first
        if not check_dependencies():
            logger.error("[FAIL] Required dependencies missing")
            sys.exit(1)
        
        # Parse command line arguments
        args = parse_command_line_arguments()
        
        # Create game configuration
        config = GameConfiguration.from_args(args)
        
        # Load configuration file if specified
        if config.config_file and os.path.exists(config.config_file):
            file_config = GameConfiguration.from_file(config.config_file)
            # Merge configurations (command line takes precedence)
            for field in config.__dataclass_fields__:
                if getattr(config, field) == getattr(GameConfiguration(), field):
                    setattr(config, field, getattr(file_config, field))
        
        # Validate configuration
        config.validate()
        
        # Force headless mode if pygame not available
        if not PYGAME_AVAILABLE:
            config.headless_mode = True
            config.show_debug_overlay = False
        
        # Validate system requirements if threading enabled
        if config.use_threading:
            if not validate_system_requirements():
                logger.warning("[WARN] System validation failed, forcing --no-threading mode")
                config.use_threading = False
        
        # Configuration summary
        logger.info("[OK] Configuration:")
        logger.info(f"  Threading: {'enabled' if config.use_threading else 'disabled'}")
        logger.info(f"  Resolution: {config.width}x{config.height}")
        logger.info(f"  FPS: {config.fps}")
        logger.info(f"  Debug: {'enabled' if config.debug_mode else 'disabled'}")
        logger.info(f"  Headless: {'enabled' if config.headless_mode else 'disabled'}")
        
        if config.test_duration > 0:
            logger.info(f"  Test mode: {config.test_duration} seconds")
        
        # Create and run game loop
        game_loop = GameLoop(config)
        success = game_loop.run()
        
        # Final status
        if success:
            logger.info("[OK] Simulation completed successfully")
        else:
            logger.error("[FAIL] Simulation completed with errors")
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("[WARN] Simulation terminated by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"[FAIL] Fatal error: {e}")
        if '--debug' in sys.argv or '-debug' in sys.argv:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()