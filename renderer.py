# renderer.py
"""
Complete Rendering System Interface for Solar Farm Cloud Simulation

Provides rendering interface for all simulation components with CSV coordinate integration:
- Solar panel arrays with real-time power visualization  
- 3D cloud visualization with shadows and atmospheric effects
- Wind indicators and environmental data display
- Performance overlay and debug information with threading metrics
- Multi-backend support (Pygame, OpenGL, Console) with quality settings
- Frame management optimized for 60 FPS threading architecture
"""
import time
import math
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

# Try to import pygame for rendering
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: Pygame not available. Using console-only renderer.")

# Try to import OpenGL for advanced rendering
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import OpenGL.GL.shaders as shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Using Pygame-only renderer.")

class Renderer:
    """
    Complete rendering interface supporting multiple backends and CSV coordinate integration.
    
    Features:
    - Real-time solar panel visualization with power-based coloring
    - 3D cloud rendering with atmospheric effects
    - Shadow overlay system integrated with shadow calculator
    - Wind indicators and environmental data display
    - Performance monitoring with threading statistics
    - Multi-quality rendering (Low/Medium/High)
    - CSV coordinate system integration for accurate panel positioning
    """
    
    def __init__(self, width: int = 1024, height: int = 768, fullscreen: bool = False,
                 vsync: bool = True, quality: str = "medium"):
        """
        Initialize the renderer with display settings and coordinate system.
        
        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            fullscreen: Enable fullscreen mode
            vsync: Enable vertical sync
            quality: Rendering quality ("low", "medium", "high")
        """
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.vsync = vsync
        self.quality = quality
        
        # Rendering backend selection
        self.use_opengl = OPENGL_AVAILABLE and quality in ["medium", "high"]
        self.use_pygame = PYGAME_AVAILABLE
        self.console_only = not (PYGAME_AVAILABLE or OPENGL_AVAILABLE)
        
        # Display state
        self.screen = None
        self.clock = None
        self.running = False
        
        # Coordinate system for CSV integration
        self.coordinate_bounds = {
            'x_min': 0.0, 'x_max': 50.0,  # km
            'y_min': 0.0, 'y_max': 50.0   # km
        }
        self.auto_bounds = True  # Auto-calculate from panel data
        
        # Threading and performance
        self._lock = threading.RLock()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.render_times = []
        self.max_render_history = 60  # Keep 1 second of frame times
        
        # Rendering parameters based on quality
        self._initialize_quality_settings()
        
        # Color schemes and visual styles
        self._initialize_color_schemes()
        
        # Panel rendering optimization
        self.panel_sprites = {}  # Cache for panel rendering
        self.panel_batch_size = 100  # Batch rendering for performance
        
        # Cloud rendering system
        self.cloud_particles = []
        self.cloud_animation_time = 0.0
        
        # Performance overlay
        self.show_debug = False
        self.show_performance = False
        self.overlay_font = None
        self.overlay_surfaces = {}  # Cache overlay surfaces
        
        # Initialize display
        self._initialize_display()
        
        print(f"Renderer initialized successfully:")
        print(f"  Resolution: {width}x{height} ({'Fullscreen' if fullscreen else 'Windowed'})")
        print(f"  Backend: {'OpenGL' if self.use_opengl else 'Pygame' if self.use_pygame else 'Console'}")
        print(f"  Quality: {quality} (Detail Level: {self.panel_detail_level})")
        print(f"  VSync: {'Enabled' if vsync else 'Disabled'}")
        print(f"  Anti-aliasing: {'Enabled' if self.anti_aliasing else 'Disabled'}")
    
    def _initialize_quality_settings(self):
        """Initialize rendering parameters based on quality settings."""
        quality_configs = {
            "low": {
                'panel_detail_level': 1,
                'cloud_detail_level': 1,
                'shadow_resolution': 32,
                'anti_aliasing': False,
                'particle_count': 10,
                'shadow_softness': 0.2,
                'cloud_layers': 1
            },
            "medium": {
                'panel_detail_level': 2,
                'cloud_detail_level': 2,
                'shadow_resolution': 64,
                'anti_aliasing': True,
                'particle_count': 25,
                'shadow_softness': 0.4,
                'cloud_layers': 2
            },
            "high": {
                'panel_detail_level': 3,
                'cloud_detail_level': 3,
                'shadow_resolution': 128,
                'anti_aliasing': True,
                'particle_count': 50,
                'shadow_softness': 0.6,
                'cloud_layers': 3
            }
        }
        
        config = quality_configs.get(self.quality, quality_configs["medium"])
        for key, value in config.items():
            setattr(self, key, value)
        
        # Animation and effect parameters
        self.cloud_animation_speed = 1.0
        self.wind_particle_lifetime = 2.0
        self.panel_size_base = 6  # Base panel size in pixels
    
    def _initialize_color_schemes(self):
        """Initialize comprehensive color schemes for all rendering elements."""
        # Sky and environment colors
        self.environment_colors = {
            'sky_day': (135, 206, 235),      # Light blue
            'sky_sunset': (255, 165, 0),     # Orange
            'sky_night': (25, 25, 112),      # Midnight blue
            'ground': (34, 139, 34),         # Forest green
            'grid_major': (200, 200, 200),   # Light gray
            'grid_minor': (240, 240, 240),   # Very light gray
            'horizon': (169, 169, 169)       # Dark gray
        }
        
        # Solar panel colors based on power output and status
        self.panel_colors = {
            'excellent': (0, 255, 0),        # Bright green (>90% capacity)
            'good': (173, 255, 47),          # Green-yellow (70-90%)
            'fair': (255, 255, 0),           # Yellow (50-70%)
            'poor': (255, 165, 0),           # Orange (20-50%)
            'very_poor': (255, 69, 0),       # Red-orange (5-20%)
            'offline': (128, 128, 128),      # Gray (0-5%)
            'fault': (255, 0, 0),            # Red (fault condition)
            'shaded_light': (100, 149, 237), # Cornflower blue (10-30% shaded)
            'shaded_medium': (65, 105, 225), # Royal blue (30-60% shaded)
            'shaded_heavy': (25, 25, 112),   # Midnight blue (>60% shaded)
            'maintenance': (255, 20, 147),   # Deep pink
            'border': (0, 0, 0),             # Black border
            'highlight': (255, 255, 255)     # White highlight
        }
        
        # Cloud rendering colors with transparency
        self.cloud_colors = {
            'cumulus': (255, 255, 255, 220),    # Bright white
            'cumulus_dense': (240, 240, 240, 240), # Dense white
            'stratus': (220, 220, 220, 180),    # Light gray
            'stratus_thick': (200, 200, 200, 200), # Medium gray
            'cirrus': (240, 240, 255, 150),     # Very light blue-white
            'cirrus_wispy': (250, 250, 255, 120), # Wispy white
            'shadow_light': (0, 0, 0, 60),      # Light shadow
            'shadow_medium': (0, 0, 0, 100),    # Medium shadow
            'shadow_heavy': (0, 0, 0, 140),     # Heavy shadow
            'storm': (80, 80, 80, 200)          # Dark storm clouds
        }
        
        # Wind and weather visualization
        self.weather_colors = {
            'wind_arrow': (0, 100, 200),        # Blue wind arrow
            'wind_particles': (173, 216, 230),  # Light blue particles
            'temperature_cold': (0, 0, 255),    # Blue (cold)
            'temperature_cool': (0, 255, 255),  # Cyan (cool)
            'temperature_warm': (255, 255, 0),  # Yellow (warm)
            'temperature_hot': (255, 165, 0),   # Orange (hot)
            'temperature_extreme': (255, 0, 0), # Red (extreme)
            'humidity_low': (139, 69, 19),      # Brown (dry)
            'humidity_medium': (34, 139, 34),   # Green (normal)
            'humidity_high': (0, 0, 139)        # Dark blue (humid)
        }
        
        # UI and overlay colors
        self.ui_colors = {
            'overlay_bg': (0, 0, 0, 180),       # Semi-transparent black
            'overlay_border': (255, 255, 255, 100), # Semi-transparent white
            'text_normal': (255, 255, 255),     # White text
            'text_highlight': (255, 255, 0),    # Yellow highlight
            'text_success': (0, 255, 0),        # Green success
            'text_warning': (255, 165, 0),      # Orange warning
            'text_error': (255, 0, 0),          # Red error
            'text_info': (173, 216, 230),       # Light blue info
            'graph_line': (0, 255, 255),        # Cyan graph lines
            'graph_fill': (0, 255, 255, 60)     # Transparent cyan fill
        }
    
    def _initialize_display(self):
        """Initialize the display backend with error handling."""
        if self.console_only:
            print("Renderer: Console-only mode (no graphics libraries available)")
            self.running = True
            return
        
        if not self.use_pygame:
            print("ERROR: No suitable rendering backend available")
            return
        
        try:
            pygame.init()
            
            # Set display mode with appropriate flags
            flags = pygame.DOUBLEBUF | pygame.HWSURFACE
            if self.fullscreen:
                flags |= pygame.FULLSCREEN
            if self.use_opengl:
                flags |= pygame.OPENGL
            
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
            pygame.display.set_caption("Solar Farm Cloud Simulation - Advanced Threading")
            
            # Initialize clock for FPS control
            self.clock = pygame.time.Clock()
            
            # Initialize font system
            pygame.font.init()
            try:
                self.overlay_font = pygame.font.Font(None, 20)
                self.title_font = pygame.font.Font(None, 28)
            except:
                self.overlay_font = pygame.font.SysFont('monospace', 16)
                self.title_font = pygame.font.SysFont('monospace', 20)
            
            # Initialize OpenGL if selected
            if self.use_opengl:
                self._initialize_opengl()
            
            self.running = True
            print("✓ Display backend initialized successfully")
            
        except Exception as e:
            print(f"✗ Error initializing display: {e}")
            print("Falling back to console-only mode")
            self.console_only = True
            self.use_pygame = False
            self.use_opengl = False
            self.running = True
    
    def _initialize_opengl(self):
        """Initialize OpenGL context with enhanced settings."""
        if not self.use_opengl:
            return
        
        try:
            # Enable depth testing and blending
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Enable anti-aliasing if requested
            if self.anti_aliasing:
                glEnable(GL_MULTISAMPLE)
                glEnable(GL_LINE_SMOOTH)
                glEnable(GL_POLYGON_SMOOTH)
                glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
                glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
            
            # Set viewport
            glViewport(0, 0, self.width, self.height)
            
            # Configure projection matrix
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, self.width / self.height, 0.1, 1000.0)
            
            # Configure modelview matrix
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Set clear color to sky blue
            glClearColor(0.53, 0.81, 0.92, 1.0)
            
            print("✓ OpenGL context initialized with enhanced settings")
            
        except Exception as e:
            print(f"✗ Error initializing OpenGL: {e}")
            print("Falling back to Pygame rendering")
            self.use_opengl = False
    
    def update_coordinate_bounds(self, panel_df: Optional[pd.DataFrame]):
        """Update coordinate bounds based on panel data for accurate rendering."""
        if panel_df is None or len(panel_df) == 0 or not self.auto_bounds:
            return
        
        try:
            # Calculate bounds with margin
            margin_km = 2.0  # 2km margin around panels
            
            self.coordinate_bounds = {
                'x_min': float(panel_df['x_km'].min()) - margin_km,
                'x_max': float(panel_df['x_km'].max()) + margin_km,
                'y_min': float(panel_df['y_km'].min()) - margin_km,
                'y_max': float(panel_df['y_km'].max()) + margin_km
            }
            
            print(f"Updated coordinate bounds: X={self.coordinate_bounds['x_min']:.1f}-{self.coordinate_bounds['x_max']:.1f} km, "
                  f"Y={self.coordinate_bounds['y_min']:.1f}-{self.coordinate_bounds['y_max']:.1f} km")
            
        except Exception as e:
            print(f"Error updating coordinate bounds: {e}")
    
    def _convert_coordinates(self, x_km: float, y_km: float) -> Tuple[int, int]:
        """Convert CSV coordinates (km) to screen coordinates (pixels)."""
        # Normalize coordinates to 0-1 range
        x_norm = (x_km - self.coordinate_bounds['x_min']) / (self.coordinate_bounds['x_max'] - self.coordinate_bounds['x_min'])
        y_norm = (y_km - self.coordinate_bounds['y_min']) / (self.coordinate_bounds['y_max'] - self.coordinate_bounds['y_min'])
        
        # Convert to screen coordinates
        screen_x = int(x_norm * self.width)
        screen_y = int((1.0 - y_norm) * self.height)  # Flip Y axis for screen coordinates
        
        return screen_x, screen_y
    
    def clear(self):
        """Clear the screen for new frame with environment-appropriate background."""
        if self.console_only:
            return
        
        self.frame_count += 1
        
        if self.use_opengl:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # Sky color changes based on time (simplified day/night cycle)
            glClearColor(0.53, 0.81, 0.92, 1.0)  # Day sky
        elif self.use_pygame:
            # Fill with appropriate sky color
            self.screen.fill(self.environment_colors['sky_day'])
            
            # Draw coordinate grid if enabled
            if self.panel_detail_level >= 2:
                self._draw_coordinate_grid()
    
    def _draw_coordinate_grid(self):
        """Draw coordinate grid for reference."""
        if not self.use_pygame:
            return
        
        try:
            # Draw major grid lines (every 10 km)
            grid_spacing_major = 10.0  # km
            grid_spacing_minor = 5.0   # km
            
            # Calculate grid lines
            x_start = int(self.coordinate_bounds['x_min'] / grid_spacing_major) * grid_spacing_major
            x_end = int(self.coordinate_bounds['x_max'] / grid_spacing_major + 1) * grid_spacing_major
            y_start = int(self.coordinate_bounds['y_min'] / grid_spacing_major) * grid_spacing_major
            y_end = int(self.coordinate_bounds['y_max'] / grid_spacing_major + 1) * grid_spacing_major
            
            # Draw vertical lines
            for x_km in np.arange(x_start, x_end, grid_spacing_major):
                screen_x, _ = self._convert_coordinates(x_km, self.coordinate_bounds['y_min'])
                if 0 <= screen_x <= self.width:
                    pygame.draw.line(self.screen, self.environment_colors['grid_major'],
                                   (screen_x, 0), (screen_x, self.height), 1)
            
            # Draw horizontal lines
            for y_km in np.arange(y_start, y_end, grid_spacing_major):
                _, screen_y = self._convert_coordinates(self.coordinate_bounds['x_min'], y_km)
                if 0 <= screen_y <= self.height:
                    pygame.draw.line(self.screen, self.environment_colors['grid_major'],
                                   (0, screen_y), (self.width, screen_y), 1)
                                   
        except Exception as e:
            pass  # Grid drawing is optional
    
    def present(self):
        """Present the current frame and update performance metrics."""
        if self.console_only:
            return
        
        if self.use_pygame:
            pygame.display.flip()
        
        # Update FPS counter and performance metrics
        current_time = time.time()
        frame_time = current_time - self.last_fps_time
        
        # Update render time history
        self.render_times.append(frame_time)
        if len(self.render_times) > self.max_render_history:
            self.render_times.pop(0)
        
        # Calculate FPS every second
        if len(self.render_times) >= 30:  # Calculate every 30 frames for stability
            self.current_fps = len(self.render_times) / sum(self.render_times)
            
        self.last_fps_time = current_time
    
    def render_solar_panels(self, panel_df: pd.DataFrame, panel_coverage: Dict[str, float],
                          power_output: Dict[str, Any], x_range: Tuple[float, float] = None,
                          y_range: Tuple[float, float] = None):
        """
        Render solar panel array with comprehensive power and status visualization.
        
        Args:
            panel_df: Panel DataFrame with CSV coordinates
            panel_coverage: Panel cloud coverage dictionary  
            power_output: Complete power output data from power simulator
            x_range: X coordinate range (deprecated - uses auto bounds)
            y_range: Y coordinate range (deprecated - uses auto bounds)
        """
        if self.console_only or panel_df is None or len(panel_df) == 0:
            if self.console_only:
                self._render_panels_console(panel_df, panel_coverage, power_output)
            return
        
        try:
            # Update coordinate bounds from panel data
            self.update_coordinate_bounds(panel_df)
            
            if self.use_opengl:
                self._render_solar_panels_opengl(panel_df, panel_coverage, power_output)
            else:
                self._render_solar_panels_pygame(panel_df, panel_coverage, power_output)
                
        except Exception as e:
            print(f"Error rendering solar panels: {e}")
    
    def _render_panels_console(self, panel_df: pd.DataFrame, panel_coverage: Dict[str, float],
                             power_output: Dict[str, Any]):
        """Render panel information to console."""
        if panel_df is None:
            return
        
        total_power = power_output.get('total', 0.0)
        total_panels = len(panel_df)
        shadowed_panels = sum(1 for cov in panel_coverage.values() if cov > 0.1)
        
        print(f"☀️  SOLAR FARM STATUS: {total_power:.1f}kW total, "
              f"{shadowed_panels}/{total_panels} panels shadowed")
    
    def _render_solar_panels_pygame(self, panel_df: pd.DataFrame, panel_coverage: Dict[str, float],
                                   power_output: Dict[str, Any]):
        """Render solar panels using Pygame with enhanced visualization."""
        for _, row in panel_df.iterrows():
            panel_id = row['panel_id']
            x_km = row['x_km']
            y_km = row['y_km']
            
            # Convert coordinates to screen space
            screen_x, screen_y = self._convert_coordinates(x_km, y_km)
            
            # Skip panels outside screen bounds
            if not (0 <= screen_x <= self.width and 0 <= screen_y <= self.height):
                continue
            
            # Determine panel color and size based on status
            color, border_color, panel_size = self._get_panel_visual_properties(
                panel_id, panel_coverage, power_output
            )
            
            # Draw panel with quality-dependent details
            self._draw_panel_pygame(screen_x, screen_y, panel_size, color, border_color, panel_id, 
                                  panel_coverage.get(panel_id, 0.0), power_output)
    
    def _draw_panel_pygame(self, screen_x: int, screen_y: int, size: int, color: Tuple[int, int, int],
                         border_color: Tuple[int, int, int], panel_id: str, coverage: float,
                         power_output: Dict[str, Any]):
        """Draw individual panel with quality-based detail levels."""
        half_size = size // 2
        panel_rect = (screen_x - half_size, screen_y - half_size, size, size)
        
        # Draw main panel
        pygame.draw.rect(self.screen, color, panel_rect)
        
        # Draw border for medium+ quality
        if self.panel_detail_level >= 2:
            pygame.draw.rect(self.screen, border_color, panel_rect, 1)
        
        # Draw shadow overlay if panel is covered
        if coverage > 0.05:
            shadow_alpha = min(200, int(coverage * 255 * 0.8))
            shadow_color = (*self.cloud_colors['shadow_medium'][:3], shadow_alpha)
            
            # Create shadow surface
            shadow_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            shadow_surface.fill(shadow_color)
            self.screen.blit(shadow_surface, (screen_x - half_size, screen_y - half_size))
        
        # Draw power indicator for high quality
        if self.panel_detail_level >= 3 and panel_id in power_output:
            panel_data = power_output.get(panel_id, {})
            if isinstance(panel_data, dict):
                self._draw_power_indicator(screen_x, screen_y, size, panel_data)
    
    def _draw_power_indicator(self, x: int, y: int, size: int, panel_data: Dict[str, Any]):
        """Draw power output indicator for high-detail mode."""
        try:
            power = panel_data.get('final_power', 0.0)
            capacity = panel_data.get('capacity', 400.0)
            power_ratio = power / capacity if capacity > 0 else 0.0
            
            # Draw power bar
            bar_width = max(2, size // 3)
            bar_height = int(power_ratio * size)
            
            if bar_height > 1:
                bar_rect = (x + size//3, y + size//2 - bar_height//2, bar_width, bar_height)
                bar_color = (0, int(255 * power_ratio), 0)  # Green intensity based on power
                pygame.draw.rect(self.screen, bar_color, bar_rect)
                
        except Exception:
            pass  # Power indicator is optional
    
    def _render_solar_panels_opengl(self, panel_df: pd.DataFrame, panel_coverage: Dict[str, float],
                                   power_output: Dict[str, Any]):
        """Render solar panels using OpenGL for 3D visualization."""
        glBegin(GL_QUADS)
        
        for _, row in panel_df.iterrows():
            panel_id = row['panel_id']
            x_km = row['x_km']
            y_km = row['y_km']
            
            # Convert to OpenGL normalized coordinates
            x_norm = (x_km - self.coordinate_bounds['x_min']) / (self.coordinate_bounds['x_max'] - self.coordinate_bounds['x_min'])
            y_norm = (y_km - self.coordinate_bounds['y_min']) / (self.coordinate_bounds['y_max'] - self.coordinate_bounds['y_min'])
            
            gl_x = x_norm * 2.0 - 1.0  # Convert to -1 to 1 range
            gl_y = y_norm * 2.0 - 1.0
            
            # Get panel color
            color, _, _ = self._get_panel_visual_properties(panel_id, panel_coverage, power_output)
            glColor4f(color[0]/255.0, color[1]/255.0, color[2]/255.0, 1.0)
            
            # Draw panel quad
            panel_size = 0.01 * self.panel_detail_level
            glVertex3f(gl_x - panel_size, gl_y - panel_size, 0.0)
            glVertex3f(gl_x + panel_size, gl_y - panel_size, 0.0)
            glVertex3f(gl_x + panel_size, gl_y + panel_size, 0.0)
            glVertex3f(gl_x - panel_size, gl_y + panel_size, 0.0)
        
        glEnd()
    
    def _get_panel_visual_properties(self, panel_id: str, panel_coverage: Dict[str, float],
                                   power_output: Dict[str, Any]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], int]:
        """Determine panel visual properties based on power output, coverage, and status."""
        coverage = panel_coverage.get(panel_id, 0.0)
        
        # Get panel power data
        panel_data = power_output.get(panel_id, {})
        if isinstance(panel_data, dict):
            power = panel_data.get('final_power', 0.0)
            capacity = panel_data.get('capacity', 400.0)
            status = panel_data.get('status', 'normal')
        else:
            power = 0.0
            capacity = 400.0
            status = 'normal'
        
        # Calculate power ratio
        power_ratio = power / capacity if capacity > 0 else 0.0
        
        # Determine panel size based on detail level
        base_size = self.panel_size_base + self.panel_detail_level * 2
        panel_size = max(4, base_size)
        
        # Determine color based on status and performance
        if status == 'fault':
            color = self.panel_colors['fault']
            border_color = self.panel_colors['border']
        elif status == 'maintenance':
            color = self.panel_colors['maintenance']
            border_color = self.panel_colors['highlight']
        elif coverage > 0.6:
            color = self.panel_colors['shaded_heavy']
            border_color = self.panel_colors['border']
        elif coverage > 0.3:
            color = self.panel_colors['shaded_medium']
            border_color = self.panel_colors['border']
        elif coverage > 0.1:
            color = self.panel_colors['shaded_light']
            border_color = self.panel_colors['border']
        elif power_ratio > 0.9:
            color = self.panel_colors['excellent']
            border_color = self.panel_colors['border']
        elif power_ratio > 0.7:
            color = self.panel_colors['good']
            border_color = self.panel_colors['border']
        elif power_ratio > 0.5:
            color = self.panel_colors['fair']
            border_color = self.panel_colors['border']
        elif power_ratio > 0.2:
            color = self.panel_colors['poor']
            border_color = self.panel_colors['border']
        elif power_ratio > 0.05:
            color = self.panel_colors['very_poor']
            border_color = self.panel_colors['border']
        else:
            color = self.panel_colors['offline']
            border_color = self.panel_colors['border']
        
        return color, border_color, panel_size
    
    def render_clouds(self, cloud_ellipses: List, alpha: float = 0.7):
        """
        Render 3D cloud visualization with atmospheric effects.
        
        Args:
            cloud_ellipses: List of cloud ellipse data from cloud simulation
            alpha: Overall cloud transparency multiplier
        """
        if self.console_only:
            if cloud_ellipses:
                print(f"☁️  CLOUDS: {len(cloud_ellipses)} active clouds")
            return
        
        if not cloud_ellipses:
            return
        
        try:
            self.cloud_animation_time += 0.016 * self.cloud_animation_speed  # ~60 FPS increment
            
            if self.use_opengl:
                self._render_clouds_opengl(cloud_ellipses, alpha)
            else:
                self._render_clouds_pygame(cloud_ellipses, alpha)
                
        except Exception as e:
            print(f"Error rendering clouds: {e}")
    
    def _render_clouds_pygame(self, cloud_ellipses: List, alpha: float):
        """Render clouds using Pygame with layered atmospheric effects."""
        # Create cloud surface for proper alpha blending
        cloud_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Sort clouds by altitude for proper depth rendering
        sorted_clouds = sorted(cloud_ellipses, key=lambda c: c[6] if len(c) > 6 else 1000.0, reverse=True)
        
        for cloud_idx, ellipse_data in enumerate(sorted_clouds):
            if len(ellipse_data) < 6:
                continue
            
            try:
                x, y, width, height, rotation, opacity = ellipse_data[:6]
                altitude = ellipse_data[6] if len(ellipse_data) > 6 else 1000.0
                
                # Convert domain coordinates to screen coordinates
                screen_x, screen_y = self._convert_coordinates(x / 1000.0, y / 1000.0)  # Convert m to km
                screen_width = int(width / 1000.0 / (self.coordinate_bounds['x_max'] - self.coordinate_bounds['x_min']) * self.width)
                screen_height = int(height / 1000.0 / (self.coordinate_bounds['y_max'] - self.coordinate_bounds['y_min']) * self.height)
                
                # Skip clouds that are too small or outside screen
                if screen_width < 5 or screen_height < 5:
                    continue
                
                # Calculate cloud color with atmospheric perspective
                cloud_alpha = self._calculate_cloud_alpha(opacity, alpha, altitude)
                
                # Determine cloud type and color
                cloud_color = self._get_cloud_color(altitude, opacity, cloud_alpha)
                
                # Draw cloud with multiple layers for depth
                self._draw_cloud_layers_pygame(cloud_surface, screen_x, screen_y, 
                                             screen_width, screen_height, cloud_color, rotation)
                
            except Exception as e:
                continue  # Skip problematic clouds
        
        # Blit cloud surface to main screen
        self.screen.blit(cloud_surface, (0, 0))
    
    def _calculate_cloud_alpha(self, opacity: float, alpha: float, altitude: float) -> int:
        """Calculate cloud alpha based on opacity, global alpha, and atmospheric perspective."""
        # Apply atmospheric perspective - higher clouds are more transparent
        altitude_factor = max(0.3, 1.0 - (altitude - 500.0) / 2500.0)  # Fade above 3km
        effective_opacity = opacity * alpha * altitude_factor
        return int(min(255, max(0, effective_opacity * 255)))
    
    def _get_cloud_color(self, altitude: float, opacity: float, cloud_alpha: int) -> Tuple[int, int, int, int]:
        """Determine cloud color based on altitude and density."""
        if altitude > 2000:  # High altitude - cirrus
            base_color = self.cloud_colors['cirrus']
        elif opacity > 0.8:  # Dense cloud
            base_color = self.cloud_colors['cumulus_dense']
        elif opacity > 0.5:  # Medium density
            base_color = self.cloud_colors['cumulus']
        else:  # Light cloud
            base_color = self.cloud_colors['stratus']
        
        return (*base_color[:3], cloud_alpha)
    
    def _draw_cloud_layers_pygame(self, surface: pygame.Surface, x: int, y: int, 
                                width: int, height: int, color: Tuple[int, int, int, int], rotation: float):
        """Draw cloud with multiple layers for realistic appearance."""
        try:
            # Main cloud body
            main_rect = (x - width//2, y - height//2, width, height)
            pygame.draw.ellipse(surface, color, main_rect)
            
            # Add detail layers for higher quality
            if self.cloud_detail_level >= 2:
                # Secondary layer (smaller, offset)
                offset_x = int(width * 0.1 * math.sin(self.cloud_animation_time + x * 0.01))
                offset_y = int(height * 0.1 * math.cos(self.cloud_animation_time + y * 0.01))
                
                secondary_color = (*color[:3], color[3] // 2)
                secondary_rect = (x - width//3 + offset_x, y - height//3 + offset_y, 
                                width//2, height//2)
                pygame.draw.ellipse(surface, secondary_color, secondary_rect)
            
            if self.cloud_detail_level >= 3:
                # Tertiary detail layer
                detail_color = (*color[:3], color[3] // 3)
                detail_rect = (x - width//4, y - height//4, width//3, height//3)
                pygame.draw.ellipse(surface, detail_color, detail_rect)
                
        except Exception:
            pass  # Cloud layer drawing is optional
    
    def _render_clouds_opengl(self, cloud_ellipses: List, alpha: float):
        """Render clouds using OpenGL with 3D atmospheric effects."""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        for ellipse_data in cloud_ellipses:
            if len(ellipse_data) < 6:
                continue
            
            x, y, width, height, rotation, opacity = ellipse_data[:6]
            altitude = ellipse_data[6] if len(ellipse_data) > 6 else 1000.0
            
            # Convert to OpenGL coordinates
            gl_x = (x / 1000.0 - self.coordinate_bounds['x_min']) / (self.coordinate_bounds['x_max'] - self.coordinate_bounds['x_min']) * 2.0 - 1.0
            gl_y = (y / 1000.0 - self.coordinate_bounds['y_min']) / (self.coordinate_bounds['y_max'] - self.coordinate_bounds['y_min']) * 2.0 - 1.0
            gl_z = altitude / 5000.0  # Scale altitude
            gl_width = width / 25000.0
            gl_height = height / 25000.0
            
            # Set cloud color with atmospheric perspective
            cloud_alpha = opacity * alpha * max(0.3, 1.0 - altitude / 3000.0)
            glColor4f(1.0, 1.0, 1.0, cloud_alpha)
            
            # Draw cloud as textured polygon
            segments = 8 + self.cloud_detail_level * 4
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(gl_x, gl_y, gl_z)  # Center
            
            for i in range(segments + 1):
                angle = 2.0 * math.pi * i / segments
                px = gl_x + gl_width * math.cos(angle)
                py = gl_y + gl_height * math.sin(angle)
                glVertex3f(px, py, gl_z)
            
            glEnd()
    
    def render_shadows(self, panel_coverage: Dict[str, float], x_range: Tuple[float, float] = None,
                      y_range: Tuple[float, float] = None):
        """
        Render shadow overlay on panels (integrated into panel rendering for efficiency).
        
        Args:
            panel_coverage: Panel shadow coverage data
            x_range: Deprecated - using auto bounds
            y_range: Deprecated - using auto bounds
        """
        # Shadow rendering is integrated into render_solar_panels for efficiency
        # This method maintains API compatibility
        pass
    
    def render_wind_indicators(self, wind_speed: float, wind_direction: float,
                             position: Optional[Tuple[float, float]] = None):
        """
        Render wind direction and speed indicators with particles.
        
        Args:
            wind_speed: Wind speed in m/s
            wind_direction: Wind direction in degrees (0=north, 90=east)
            position: Screen position for indicator (None for default corner)
        """
        if self.console_only:
            print(f"💨 WIND: {wind_speed:.1f} m/s @ {wind_direction:.0f}°")
            return
        
        try:
            if position is None:
                # Default position in top-right corner
                center_x = self.width - 100
                center_y = 80
            else:
                center_x, center_y = position
            
            if self.use_pygame:
                self._render_wind_indicator_pygame(center_x, center_y, wind_speed, wind_direction)
                
                # Render wind particles for enhanced visualization
                if self.panel_detail_level >= 2:
                    self._render_wind_particles_pygame(wind_speed, wind_direction)
                    
        except Exception as e:
            print(f"Error rendering wind indicator: {e}")
    
    def _render_wind_indicator_pygame(self, center_x: int, center_y: int, 
                                    wind_speed: float, wind_direction: float):
        """Render enhanced wind indicator with compass and speed visualization."""
        radius = 35
        
        # Draw compass background circle
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), radius, 2)
        pygame.draw.circle(self.screen, (0, 0, 0, 100), (center_x, center_y), radius - 1)
        
        # Draw cardinal direction markers
        for angle, label in [(0, 'N'), (90, 'E'), (180, 'S'), (270, 'W')]:
            marker_angle = math.radians(angle)
            marker_x = center_x + (radius - 10) * math.sin(marker_angle)
            marker_y = center_y - (radius - 10) * math.cos(marker_angle)
            
            if self.overlay_font:
                text = self.overlay_font.render(label, True, self.ui_colors['text_normal'])
                text_rect = text.get_rect(center=(marker_x, marker_y))
                self.screen.blit(text, text_rect)
        
        # Draw wind arrow with speed-based length and color
        max_arrow_length = radius - 15
        arrow_length = min(max_arrow_length, wind_speed * 4)  # Scale with wind speed
        
        # Convert wind direction to screen angle (wind direction is "from" direction)
        angle_rad = math.radians(wind_direction)
        
        end_x = center_x + arrow_length * math.sin(angle_rad)
        end_y = center_y - arrow_length * math.cos(angle_rad)
        
        # Color based on wind speed
        if wind_speed < 2:
            arrow_color = (100, 100, 255)  # Light blue - calm
        elif wind_speed < 5:
            arrow_color = (0, 150, 255)    # Blue - light
        elif wind_speed < 10:
            arrow_color = (0, 255, 0)      # Green - moderate
        elif wind_speed < 15:
            arrow_color = (255, 255, 0)    # Yellow - strong
        else:
            arrow_color = (255, 100, 0)    # Orange - very strong
        
        # Draw arrow shaft
        pygame.draw.line(self.screen, arrow_color, (center_x, center_y), (end_x, end_y), 4)
        
        # Draw arrow head
        if arrow_length > 10:
            head_length = 10
            head_angle = math.pi / 6  # 30 degrees
            
            head1_x = end_x - head_length * math.sin(angle_rad + head_angle)
            head1_y = end_y + head_length * math.cos(angle_rad + head_angle)
            head2_x = end_x - head_length * math.sin(angle_rad - head_angle)
            head2_y = end_y + head_length * math.cos(angle_rad - head_angle)
            
            pygame.draw.polygon(self.screen, arrow_color, 
                              [(end_x, end_y), (head1_x, head1_y), (head2_x, head2_y)])
        
        # Draw wind speed text with background
        if self.overlay_font:
            speed_text = f"{wind_speed:.1f} m/s"
            text_surface = self.overlay_font.render(speed_text, True, self.ui_colors['text_normal'])
            text_rect = text_surface.get_rect(center=(center_x, center_y + radius + 25))
            
            # Draw text background
            bg_rect = text_rect.inflate(10, 4)
            pygame.draw.rect(self.screen, (0, 0, 0, 150), bg_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), bg_rect, 1)
            
            self.screen.blit(text_surface, text_rect)
    
    def _render_wind_particles_pygame(self, wind_speed: float, wind_direction: float):
        """Render animated wind particles across the screen."""
        # Update and render wind particles
        if not hasattr(self, 'wind_particles'):
            self.wind_particles = []
        
        # Add new particles
        if len(self.wind_particles) < self.particle_count:
            for _ in range(min(3, self.particle_count - len(self.wind_particles))):
                particle = {
                    'x': float(np.random.randint(0, self.width)),
                    'y': float(np.random.randint(0, self.height)),
                    'age': 0.0,
                    'lifetime': self.wind_particle_lifetime
                }
                self.wind_particles.append(particle)
        
        # Update and draw particles
        dt = 0.016  # Assume 60 FPS
        wind_rad = math.radians(wind_direction)
        wind_vel_x = wind_speed * 5 * math.sin(wind_rad)  # Scale for visibility
        wind_vel_y = -wind_speed * 5 * math.cos(wind_rad)
        
        active_particles = []
        for particle in self.wind_particles:
            particle['age'] += dt
            
            if particle['age'] < particle['lifetime']:
                # Update position
                particle['x'] += wind_vel_x * dt
                particle['y'] += wind_vel_y * dt
                
                # Wrap around screen
                if particle['x'] < 0:
                    particle['x'] = self.width
                elif particle['x'] > self.width:
                    particle['x'] = 0
                if particle['y'] < 0:
                    particle['y'] = self.height
                elif particle['y'] > self.height:
                    particle['y'] = 0
                
                # Draw particle
                alpha_factor = 1.0 - (particle['age'] / particle['lifetime'])
                particle_alpha = int(alpha_factor * 150)
                particle_color = (*self.weather_colors['wind_particles'], particle_alpha)
                
                particle_surface = pygame.Surface((3, 3), pygame.SRCALPHA)
                particle_surface.fill(particle_color)
                self.screen.blit(particle_surface, (int(particle['x']), int(particle['y'])))
                
                active_particles.append(particle)
        
        self.wind_particles = active_particles
    
    def render_debug_overlay(self, debug_info: Dict[str, Any]):
        """
        Render debug information overlay with enhanced formatting.
        
        Args:
            debug_info: Dictionary with debug information
        """
        if self.console_only:
            print("🔧 DEBUG:", debug_info)
            return
        
        if not self.use_pygame or not self.overlay_font:
            return
        
        try:
            overlay_width = 350
            overlay_height = len(debug_info) * 22 + 30
            overlay_x = 10
            overlay_y = 10
            
            # Create overlay surface with border
            overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
            overlay_surface.fill(self.ui_colors['overlay_bg'])
            pygame.draw.rect(overlay_surface, self.ui_colors['overlay_border'], 
                           overlay_surface.get_rect(), 2)
            
            # Title
            if self.title_font:
                title_text = self.title_font.render("DEBUG INFO", True, self.ui_colors['text_highlight'])
                overlay_surface.blit(title_text, (10, 8))
            
            # Debug data
            y_offset = 35
            for key, value in debug_info.items():
                # Format value based on type
                if isinstance(value, float):
                    if key.lower().contains('fps'):
                        formatted_value = f"{value:.1f}"
                    else:
                        formatted_value = f"{value:.2f}"
                elif isinstance(value, bool):
                    formatted_value = "✓" if value else "✗"
                else:
                    formatted_value = str(value)
                
                # Color based on content
                if "error" in key.lower() or "fault" in key.lower():
                    text_color = self.ui_colors['text_error']
                elif "warning" in key.lower():
                    text_color = self.ui_colors['text_warning']
                elif "✓" in formatted_value or "running" in str(value).lower():
                    text_color = self.ui_colors['text_success']
                else:
                    text_color = self.ui_colors['text_normal']
                
                text = f"{key}: {formatted_value}"
                text_surface = self.overlay_font.render(text, True, text_color)
                overlay_surface.blit(text_surface, (10, y_offset))
                y_offset += 22
            
            # Blit overlay to screen
            self.screen.blit(overlay_surface, (overlay_x, overlay_y))
            
        except Exception as e:
            print(f"Error rendering debug overlay: {e}")
    
    def render_performance_overlay(self, perf_info: Dict[str, Any]):
        """
        Render comprehensive performance statistics overlay.
        
        Args:
            perf_info: Dictionary with performance metrics from threading system
        """
        if self.console_only:
            print("📊 PERFORMANCE:", perf_info)
            return
        
        if not self.use_pygame or not self.overlay_font:
            return
        
        try:
            overlay_width = 400
            overlay_height = len(perf_info) * 22 + 80  # Extra space for graphs
            overlay_x = self.width - overlay_width - 10
            overlay_y = 10
            
            # Create overlay surface
            overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
            overlay_surface.fill(self.ui_colors['overlay_bg'])
            pygame.draw.rect(overlay_surface, self.ui_colors['overlay_border'], 
                           overlay_surface.get_rect(), 2)
            
            # Title
            if self.title_font:
                title_text = self.title_font.render("PERFORMANCE MONITOR", True, self.ui_colors['text_highlight'])
                overlay_surface.blit(title_text, (10, 8))
            
            # Performance metrics
            y_offset = 35
            for key, value in perf_info.items():
                # Format and color code values
                formatted_value, text_color = self._format_performance_value(key, value)
                
                text = f"{key}: {formatted_value}"
                text_surface = self.overlay_font.render(text, True, text_color)
                overlay_surface.blit(text_surface, (10, y_offset))
                y_offset += 22
            
            # Render FPS graph if we have frame time data
            if self.render_times and len(self.render_times) > 10:
                self._draw_fps_graph(overlay_surface, 10, y_offset + 10, overlay_width - 20, 40)
            
            # Blit overlay to screen
            self.screen.blit(overlay_surface, (overlay_x, overlay_y))
            
        except Exception as e:
            print(f"Error rendering performance overlay: {e}")
    
    def _format_performance_value(self, key: str, value: Any) -> Tuple[str, Tuple[int, int, int]]:
        """Format performance values with appropriate colors."""
        key_lower = key.lower()
        
        if isinstance(value, float):
            if "fps" in key_lower:
                formatted = f"{value:.1f}"
                if value >= 50:
                    color = self.ui_colors['text_success']
                elif value >= 30:
                    color = self.ui_colors['text_warning']
                else:
                    color = self.ui_colors['text_error']
            elif "pressure" in key_lower or "queue" in key_lower:
                formatted = f"{value:.2f}"
                if value < 0.6:
                    color = self.ui_colors['text_success']
                elif value < 0.8:
                    color = self.ui_colors['text_warning']
                else:
                    color = self.ui_colors['text_error']
            elif "time" in key_lower and "ms" not in key_lower:
                formatted = f"{value:.1f}s"
                color = self.ui_colors['text_normal']
            elif "ms" in key_lower or "time" in key_lower:
                formatted = f"{value:.2f}ms"
                if value < 16:
                    color = self.ui_colors['text_success']
                elif value < 33:
                    color = self.ui_colors['text_warning']
                else:
                    color = self.ui_colors['text_error']
            else:
                formatted = f"{value:.2f}"
                color = self.ui_colors['text_normal']
        elif isinstance(value, int):
            if "error" in key_lower or "drop" in key_lower:
                formatted = str(value)
                color = self.ui_colors['text_error'] if value > 0 else self.ui_colors['text_success']
            else:
                formatted = str(value)
                color = self.ui_colors['text_normal']
        elif isinstance(value, bool):
            formatted = "✓ Yes" if value else "✗ No"
            color = self.ui_colors['text_success'] if value else self.ui_colors['text_error']
        else:
            formatted = str(value)
            if "running" in formatted.lower() or "enabled" in formatted.lower():
                color = self.ui_colors['text_success']
            elif "error" in formatted.lower() or "failed" in formatted.lower():
                color = self.ui_colors['text_error']
            else:
                color = self.ui_colors['text_normal']
        
        return formatted, color
    
    def _draw_fps_graph(self, surface: pygame.Surface, x: int, y: int, width: int, height: int):
        """Draw real-time FPS graph."""
        try:
            # Calculate FPS values from frame times
            fps_values = [1.0/max(0.001, ft) for ft in self.render_times[-min(60, len(self.render_times)):]]
            
            if len(fps_values) < 2:
                return
            
            # Draw graph background
            graph_rect = (x, y, width, height)
            pygame.draw.rect(surface, (20, 20, 20), graph_rect)
            pygame.draw.rect(surface, self.ui_colors['overlay_border'], graph_rect, 1)
            
            # Calculate scale
            max_fps = max(fps_values)
            min_fps = min(fps_values)
            fps_range = max(1, max_fps - min_fps)
            
            # Draw FPS line
            points = []
            for i, fps in enumerate(fps_values):
                px = x + (i / (len(fps_values) - 1)) * width
                py = y + height - ((fps - min_fps) / fps_range) * height
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(surface, self.ui_colors['graph_line'], False, points, 2)
            
            # Draw target FPS line (60 FPS)
            if min_fps < 60 < max_fps:
                target_y = y + height - ((60 - min_fps) / fps_range) * height
                pygame.draw.line(surface, (255, 0, 0), (x, target_y), (x + width, target_y), 1)
            
            # Labels
            if self.overlay_font:
                max_text = self.overlay_font.render(f"{max_fps:.0f}", True, self.ui_colors['text_normal'])
                surface.blit(max_text, (x + width - 30, y))
                min_text = self.overlay_font.render(f"{min_fps:.0f}", True, self.ui_colors['text_normal'])
                surface.blit(min_text, (x + width - 30, y + height - 15))
                
        except Exception:
            pass  # Graph drawing is optional
    
    def toggle_mode(self):
        """Toggle between available rendering modes."""
        if self.use_opengl and self.use_pygame:
            self.use_opengl = not self.use_opengl
            print(f"🔄 Switched to {'OpenGL' if self.use_opengl else 'Pygame'} rendering")
        else:
            print("⚠️  Cannot toggle rendering mode (only one backend available)")
    
    def set_quality(self, quality: str):
        """
        Set rendering quality level and update parameters.
        
        Args:
            quality: Quality level ("low", "medium", "high")
        """
        if quality in ["low", "medium", "high"]:
            self.quality = quality
            self._initialize_quality_settings()
            print(f"🎨 Rendering quality set to: {quality}")
        else:
            print(f"❌ Invalid quality level: {quality}")
    
    def get_render_stats(self) -> Dict[str, Any]:
        """Get comprehensive rendering statistics."""
        avg_frame_time = sum(self.render_times) / len(self.render_times) if self.render_times else 0.0
        
        return {
            'backend': 'OpenGL' if self.use_opengl else 'Pygame' if self.use_pygame else 'Console',
            'resolution': f"{self.width}x{self.height}",
            'quality': self.quality,
            'fullscreen': self.fullscreen,
            'vsync': self.vsync,
            'current_fps': self.current_fps,
            'target_fps': 60,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'frame_count': self.frame_count,
            'anti_aliasing': self.anti_aliasing,
            'panel_detail_level': self.panel_detail_level,
            'cloud_detail_level': self.cloud_detail_level,
            'coordinate_bounds': self.coordinate_bounds.copy(),
            'particle_count': getattr(self, 'particle_count', 0),
            'running': self.running
        }
    
    def cleanup(self):
        """Clean up rendering resources and shutdown display."""
        print("🧹 Cleaning up renderer...")
        
        try:
            # Clear caches
            self.panel_sprites.clear()
            self.overlay_surfaces.clear()
            if hasattr(self, 'wind_particles'):
                self.wind_particles.clear()
            
            # Shutdown pygame
            if self.use_pygame:
                pygame.quit()
            
            self.running = False
            print("✓ Renderer cleanup completed")
            
        except Exception as e:
            print(f"⚠️  Error during renderer cleanup: {e}")

# Factory functions for easy renderer creation
def create_renderer(width: int = 1024, height: int = 768, **kwargs) -> Renderer:
    """
    Factory function to create a renderer with optimal settings.
    
    Args:
        width: Screen width in pixels
        height: Screen height in pixels
        **kwargs: Additional renderer options
        
    Returns:
        Configured Renderer instance
    """
    return Renderer(width, height, **kwargs)

def create_headless_renderer() -> Renderer:
    """Create a console-only renderer for headless operation."""
    return Renderer(800, 600, fullscreen=False)

def create_high_performance_renderer(width: int = 1920, height: int = 1080) -> Renderer:
    """Create a high-performance renderer optimized for threading."""
    return Renderer(width, height, quality="high", vsync=True)

# Testing functions
def test_renderer():
    """Comprehensive renderer testing."""
    print("🧪 Testing Renderer functionality...")
    
    # Test basic renderer creation
    renderer = create_renderer(800, 600, quality="medium")
    
    if renderer.running:
        print(f"✓ Renderer initialized with {renderer.get_render_stats()['backend']} backend")
        
        # Test coordinate system
        renderer.update_coordinate_bounds(pd.DataFrame({
            'x_km': [0, 10, 20],
            'y_km': [0, 15, 30],
            'panel_id': ['P001', 'P002', 'P003']
        }))
        
        # Test rendering methods
        test_panel_df = pd.DataFrame({
            'panel_id': ['P001', 'P002', 'P003'],
            'x_km': [5, 10, 15],
            'y_km': [5, 10, 15]
        })
        
        test_coverage = {'P001': 0.3, 'P002': 0.0, 'P003': 0.7}
        test_power = {
            'P001': {'final_power': 300, 'capacity': 400, 'status': 'normal'},
            'P002': {'final_power': 380, 'capacity': 400, 'status': 'normal'},
            'P003': {'final_power': 100, 'capacity': 400, 'status': 'shaded'}
        }
        
        # Test rendering
        renderer.clear()
        renderer.render_solar_panels(test_panel_df, test_coverage, test_power)
        
        # Test cloud rendering
        test_clouds = [(10000, 15000, 5000, 3000, 0, 0.6, 1200)]
        renderer.render_clouds(test_clouds)
        
        # Test overlays
        debug_info = {'Test Mode': True, 'FPS': 60.0, 'Clouds': 1}
        renderer.render_debug_overlay(debug_info)
        
        perf_info = {'Current FPS': 59.8, 'Queue Pressure': 0.3, 'Worker Alive': True}
        renderer.render_performance_overlay(perf_info)
        
        # Test wind indicators
        renderer.render_wind_indicators(7.5, 135.0)
        
        renderer.present()
        
        print("✓ All rendering tests completed successfully")
        
        # Test quality switching
        for quality in ["low", "medium", "high"]:
            renderer.set_quality(quality)
            print(f"✓ Quality set to {quality}")
        
        # Test mode toggle
        renderer.toggle_mode()
        
    else:
        print("✗ Renderer initialization failed")
    
    # Test performance statistics
    stats = renderer.get_render_stats()
    print(f"📊 Render stats: {stats}")
    
    # Cleanup
    renderer.cleanup()
    print("🧪 Renderer testing completed")

if __name__ == "__main__":
    test_renderer()