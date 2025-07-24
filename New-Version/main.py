# main.py
#!/usr/bin/env python3
"""
Solar Farm Cloud Simulation - Optimized Main Entry Point
Streamlined Pygame visualization with efficient event handling
"""
import argparse
import time
import traceback
import pygame
import math
import numpy as np
import os
import sim_config as CFG
# from pygame_rendereres.cloud_renderer import km_to_screen_coords

# ===== CONFIGURATION DISPATCH TABLES =====
WIND_SPEED_DELTA = 0.5
MOVEMENT_DELTA = 0.5
WIND_SPEED_LIMITS = (0.5, 10.0)
MOVEMENT_LIMITS = (0.5, 3.0)

def adjust_wind_speed(delta):
    """Adjust wind speed within limits"""
    CFG.BASE_WIND_SPEED = max(WIND_SPEED_LIMITS[0], 
                             min(WIND_SPEED_LIMITS[1], 
                                 CFG.BASE_WIND_SPEED + delta))
    print(f"Wind speed: {CFG.BASE_WIND_SPEED:.1f} m/s")

def adjust_movement(delta):
    """Adjust movement multiplier within limits"""
    CFG.MOVEMENT_MULTIPLIER = max(MOVEMENT_LIMITS[0], 
                                 min(MOVEMENT_LIMITS[1], 
                                     CFG.MOVEMENT_MULTIPLIER + delta))
    print(f"Movement multiplier: {CFG.MOVEMENT_MULTIPLIER:.1f}")

def toggle_renderer(controller):
    """Toggle renderer if available"""
    if hasattr(controller, 'toggle_gl_renderer'):
        controller.toggle_gl_renderer()
    else:
        print("Renderer toggle not available")

# Event dispatch table
KEY_HANDLERS = {
    pygame.K_ESCAPE: lambda c: setattr(c, '_should_quit', True),
    pygame.K_UP: lambda c: adjust_wind_speed(WIND_SPEED_DELTA),
    pygame.K_DOWN: lambda c: adjust_wind_speed(-WIND_SPEED_DELTA),
    pygame.K_RIGHT: lambda c: adjust_movement(MOVEMENT_DELTA),
    pygame.K_LEFT: lambda c: adjust_movement(-MOVEMENT_DELTA),
    pygame.K_g: lambda c: toggle_renderer(c),
}

# ===== STREAMLINED RENDERING =====
def setup_background_fast(screen, width, height, x_range, y_range, grid_color=(204, 204, 204)):
    """Optimized background rendering with minimal calculations"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    
    # Pre-calculate grid spacing
    grid_interval_km = 5.0 if range_x > 20 else 2.0
    tick_interval = 10 if range_x > 20 else 5
    
    # Batch draw grid lines
    vlines = []
    hlines = []
    
    for x_km in np.arange(x_range[0], x_range[1] + grid_interval_km, grid_interval_km):
        x_px = int((x_km - x_range[0]) / range_x * width)
        vlines.append(((x_px, 0), (x_px, height)))
    
    for y_km in np.arange(y_range[0], y_range[1] + grid_interval_km, grid_interval_km):
        y_px = int((y_km - y_range[0]) / range_y * height)
        hlines.append(((0, y_px), (width, y_px)))
    
    # Draw all lines at once
    for start, end in vlines + hlines:
        pygame.draw.line(screen, grid_color, start, end, 1)

def create_default_panel_data(num_panels=36):
    """Fast panel data generation"""
    import pandas as pd
    grid_size = int(math.sqrt(num_panels))
    
    panel_data = [
        {
            'panel_id': f"P{i*grid_size+j+1:03d}",
            'x_km': 10 + i * 5,
            'y_km': 10 + j * 5,
            'power_capacity': 5.0
        }
        for i in range(grid_size)
        for j in range(grid_size)
        if i*grid_size+j < num_panels
    ]
    
    return pd.DataFrame(panel_data)

# ===== SIMPLIFIED CLI =====
def parse_args():
    """Minimal argument parsing for essential options only"""
    parser = argparse.ArgumentParser(description='Solar Farm Cloud Simulation')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--fps', type=int, default=CFG.FPS, help=f'FPS (default: {CFG.FPS})')
    parser.add_argument('--size', type=str, default='1200x900', help='Window size WxH (default: 1200x900)')
    parser.add_argument('--single-cloud', action='store_true', help='Single cloud mode')
    parser.add_argument('--test-gliding', action='store_true', help='Test cloud movement')
    
    args = parser.parse_args()
    
    # Parse window size
    try:
        width, height = map(int, args.size.split('x'))
    except ValueError:
        width, height = 1200, 900
        print(f"Invalid size format, using default: {width}x{height}")
    
    return args, width, height

# ===== OPTIMIZED SIMULATION LOOP =====
class OptimizedGameLoop:
    """Streamlined game loop with efficient event handling"""
    
    def __init__(self, controller, x_range, y_range, width, height, fps, debug_mode):
        self.controller = controller
        self.x_range, self.y_range = x_range, y_range
        self.width, self.height = width, height
        self.fps = fps
        self.debug_mode = debug_mode
        self.running = True
        self._should_quit = False
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Solar Farm Cloud Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        
        # Performance tracking
        self.frame_times = []
        self.cloud_positions = []
        self.frame_count = 0
        
        # Constants
        self.SKY_COLOR = (230, 242, 255)
        self.MAX_FRAME_TIMES = 100
        self.MAX_CLOUD_POSITIONS = 10
    
    def handle_events(self):
        """Optimized event handling with dispatch table"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key in KEY_HANDLERS:
                KEY_HANDLERS[event.key](self.controller)
                if hasattr(self.controller, '_should_quit'):
                    self.running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.size
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
    
    def update_simulation(self):
        """Single simulation step with error handling"""
        try:
            result = self.controller.step()
            
            # Extract cloud positions for trail (optimized)
            cloud_ellipses = result.get('cloud_ellipses', [])
            if cloud_ellipses:
                # Only track first cloud for trail
                ellipse = cloud_ellipses[0]
                x, y = ellipse[0], ellipse[1]
                km_x = x / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                km_y = y / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                
                self.cloud_positions.append((km_x, km_y))
                if len(self.cloud_positions) > self.MAX_CLOUD_POSITIONS:
                    self.cloud_positions.pop(0)
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"Simulation error: {e}")
                traceback.print_exc()
            
            # Return safe defaults
            return {
                'cloud_ellipses': [],
                'panel_coverage': {},
                'power_output': {'total': 0.0},
                'cloud_cover': 0.0,
                'alpha': 0.0
            }
    
    def render_frame(self, result):
        """Optimized rendering pipeline"""
        # Clear screen
        self.screen.fill(self.SKY_COLOR)
        
        # Background grid
        setup_background_fast(self.screen, self.width, self.height, 
                             self.x_range, self.y_range)
        
        # Solar panels
        from background_module import draw_solar_panels
        draw_solar_panels(
            self.screen, self.controller.panel_df,
            result.get('panel_coverage', {}),
            result.get('power_output', {}),
            self.x_range, self.y_range, self.width, self.height
        )
        
        # Clouds with interpolation
        self.controller.render_clouds(
            self.screen, result.get('cloud_ellipses', []), self.cloud_positions,
            self.x_range, self.y_range, self.width, self.height, 
            result.get('alpha', 0.0)
        )
        
        # UI elements
        self.controller.render_ui(self.screen, result, self.width, self.height, 
                                 self.x_range, self.y_range)
        
        # Performance overlay
        if self.debug_mode:
            self.render_debug_info(result)
        
        # FPS counter
        fps_text = f"FPS: {self.clock.get_fps():.1f}"
        fps_surface = self.font.render(fps_text, True, (0, 0, 0), (255, 255, 255, 180))
        self.screen.blit(fps_surface, (self.width - 100, 10))
    
    def render_debug_info(self, result):
        """Compact debug information display"""
        debug_lines = [
            f"Clouds: {len(result.get('cloud_ellipses', []))}",
            f"Alpha: {result.get('alpha', 0.0):.3f}",
            f"Wind: {CFG.BASE_WIND_SPEED:.1f} m/s",
            f"Coverage: {result.get('cloud_cover', 0.0):.1f}%"
        ]
        
        for i, line in enumerate(debug_lines):
            surface = self.font.render(line, True, (0, 0, 0), (255, 255, 255, 180))
            self.screen.blit(surface, (10, 10 + i * 20))
    
    def update_performance_metrics(self, frame_time):
        """Track performance efficiently"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.MAX_FRAME_TIMES:
            self.frame_times.pop(0)
        
        # Periodic debug output
        if self.debug_mode and self.frame_count % 60 == 0:
            avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
            print(f"Frame {self.frame_count}: {avg_time*1000:.1f}ms avg, "
                  f"{self.clock.get_fps():.1f} FPS")
    
    def run(self):
        """Main game loop - simplified and efficient"""
        print("Starting optimized simulation loop...")
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Handle input
                self.handle_events()
                
                # Update simulation (uses internal timestep)
                result = self.update_simulation()
                
                # Render frame
                self.render_frame(result)
                
                # Present frame
                pygame.display.flip()
                
                # Control frame rate (engine handles timing)
                self.clock.tick(self.fps)
                
                # Performance tracking
                frame_time = time.time() - frame_start
                self.update_performance_metrics(frame_time)
                self.frame_count += 1
                
        finally:
            pygame.quit()

# ===== STREAMLINED INITIALIZATION =====
def load_panel_data():
    """Simplified panel data loading"""
    panel_file = "extended_coordinates.csv"
    
    if os.path.exists(panel_file):
        try:
            from background_module import create_panel_dataframe
            panel_df = create_panel_dataframe(coordinates_path=panel_file)
            print(f"Loaded {len(panel_df)} panels from {panel_file}")
            return panel_df
        except Exception as e:
            print(f"Error loading {panel_file}: {e}")
    
    # Fallback to generated data
    print("Using generated panel data")
    return create_default_panel_data()

def initialize_controller(debug_mode, panel_df):
    """Initialize simulation controller"""
    from simulation_controller import SimulationController
    
    controller = SimulationController(
        start_time=11.0,  # Fixed start time
        timestep_minutes=5,  # Fixed timestep
        debug_mode=debug_mode,
        panel_df=panel_df
    )
    
    return controller

# ===== MAIN ENTRY POINT =====
def run_simulation():
    """Streamlined main entry point"""
    print("===== SOLAR FARM CLOUD SIMULATION =====")
    
    # Parse minimal CLI arguments
    args, width, height = parse_args()
    
    # Force cloud spawning and movement for debugging
    CFG.SPAWN_PROBABILITY = 0.8  # High probability for frequent clouds
    CFG.MIN_SPAWN_INTERVAL = 1.0  # Spawn clouds more often
    CFG.SINGLE_CLOUD_MODE = False  # Allow multiple clouds
    CFG.FORCE_INITIAL_CLOUD = True  # Always ensure at least one cloud
    
    # Apply configuration overrides
    if args.single_cloud:
        CFG.SINGLE_CLOUD_MODE = True
        print("Single cloud mode enabled")
    
    print(f"Configuration: FPS={args.fps}, Size={width}x{height}, Debug={args.debug}")
    
    try:
        # Load panel data
        panel_df = load_panel_data()
        
        # Initialize controller
        controller = initialize_controller(args.debug, panel_df)
        
        # Apply test mode if requested
        if args.test_gliding:
            print("Activating cloud gliding test...")
            # Simple test setup without complex initialization
            CFG.SINGLE_CLOUD_MODE = True
            CFG.SPAWN_PROBABILITY = 0.8  # Higher spawn rate for testing
        
        # Set simulation bounds
        x_range = (0, CFG.AREA_SIZE_KM)
        y_range = (0, CFG.AREA_SIZE_KM)
        
        # Run optimized game loop
        game_loop = OptimizedGameLoop(
            controller, x_range, y_range, 
            width, height, args.fps, args.debug
        )
        
        game_loop.run()
        
        print("Simulation completed successfully.")
        
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    except Exception as e:
        print(f"\nSimulation error: {e}")
        if args.debug:
            traceback.print_exc()

# ===== Cloud Renderer Test Block =====
if __name__ == "__main__":
    run_simulation()