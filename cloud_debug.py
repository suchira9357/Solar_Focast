# cloud_fix.py - Quick fixes for cloud appearance issues
"""
Apply fixes to ensure clouds appear in the simulation
"""

import sim_config as CFG

def apply_cloud_fixes():
    """Apply configuration fixes to ensure clouds appear"""
    print("Applying cloud appearance fixes...")
    
    # Force single cloud mode with high spawn probability
    CFG.SINGLE_CLOUD_MODE = True
    CFG.SPAWN_PROBABILITY = 0.8  # Very high spawn chance
    CFG.MAX_PARCELS = 6
    CFG.FORCE_INITIAL_CLOUD = True
    
    # Ensure proper lifecycle timing
    CFG.CLOUD_GROWTH_FRAMES = 60
    CFG.CLOUD_STABLE_FRAMES = 1800
    CFG.CLOUD_DECAY_FRAMES = 300
    
    # Set cloud appearance parameters
    CFG.MOVEMENT_MULTIPLIER = 1.0  # Normal speed
    CFG.BASE_WIND_SPEED = 4.0     # Moderate wind
    
    print("✓ Configuration fixes applied")

def create_fixed_main():
    """Create a fixed version of the main execution"""
    
    # Apply fixes first
    apply_cloud_fixes()
    
    # Import after config changes
    import pygame
    import math
    import time
    from simulation_controller import SimulationController
    from background_module import create_panel_dataframe
    
    print("Starting fixed simulation...")
    
    # Initialize pygame
    pygame.init()
    width, height = 1200, 900
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Solar Farm Cloud Simulation - FIXED")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 16)
    
    # Create panel data
    panel_df = create_panel_dataframe(num_panels=36)
    
    # Initialize controller with debug mode
    controller = SimulationController(
        start_time=11.0,
        timestep_minutes=5,
        debug_mode=True,  # Enable debug output
        panel_df=panel_df
    )
    
    # Force spawn initial clouds
    print("Force spawning initial clouds...")
    for _ in range(3):
        try:
            controller.weather_system._spawn(0)
            print(f"✓ Spawned cloud. Total: {len(controller.weather_system.parcels)}")
        except Exception as e:
            print(f"✗ Spawn error: {e}")
    
    # Set view parameters
    x_range = (0, CFG.AREA_SIZE_KM)
    y_range = (0, CFG.AREA_SIZE_KM)
    
    # Main loop
    running = True
    frame_count = 0
    
    print("Starting main loop...")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Force spawn more clouds
                    controller.weather_system._spawn(frame_count)
                    print(f"Manual spawn. Total: {len(controller.weather_system.parcels)}")
        
        # Update simulation
        try:
            result = controller.step()
        except Exception as e:
            print(f"Simulation step error: {e}")
            result = {
                'cloud_ellipses': [],
                'panel_coverage': {},
                'power_output': {'total': 0.0},
                'cloud_cover': 0.0,
                'alpha': 0.0
            }
        
        # Clear screen with sky color
        screen.fill((135, 206, 235))  # Light blue sky
        
        # Draw background grid
        from background_module import setup_background_fast
        setup_background_fast(screen, width, height, x_range, y_range)
        
        # Draw solar panels
        from background_module import draw_solar_panels
        draw_solar_panels(
            screen, controller.panel_df,
            result.get('panel_coverage', {}),
            result.get('power_output', {}),
            x_range, y_range, width, height
        )
        
        # Get cloud data
        cloud_ellipses = result.get('cloud_ellipses', [])
        cloud_positions = []
        
        # Extract cloud positions for trails
        for ellipse in cloud_ellipses:
            if len(ellipse) >= 2:
                x_m, y_m = ellipse[0], ellipse[1]
                x_km = x_m / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                y_km = y_m / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                cloud_positions.append((x_km, y_km))
        
        # Draw clouds with enhanced visibility
        if cloud_ellipses:
            try:
                controller.render_clouds(
                    screen, cloud_ellipses, cloud_positions,
                    x_range, y_range, width, height, 
                    result.get('alpha', 0.0)
                )
                print(f"✓ Rendered {len(cloud_ellipses)} clouds")
            except Exception as e:
                print(f"Cloud rendering error: {e}")
                
                # Fallback: draw simple circles for clouds
                from cloud_renderer import km_to_screen_coords
                for ellipse in cloud_ellipses:
                    if len(ellipse) >= 6:
                        x_m, y_m = ellipse[0], ellipse[1]
                        x_km = x_m / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                        y_km = y_m / CFG.DOMAIN_SIZE_M * CFG.AREA_SIZE_KM
                        opacity = ellipse[5]
                        
                        x_px, y_px = km_to_screen_coords(x_km, y_km, x_range, y_range, width, height)
                        
                        # Draw a simple white circle
                        radius = max(20, int(ellipse[2] / CFG.DOMAIN_SIZE_M * width / 10))
                        color = (255, 255, 255, int(opacity * 255))
                        
                        # Create surface for alpha blending
                        cloud_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                        pygame.draw.circle(cloud_surf, color, (radius, radius), radius)
                        screen.blit(cloud_surf, (x_px - radius, y_px - radius))
        
        # Enhanced debug information
        debug_info = [
            f"Frame: {frame_count}",
            f"Weather Parcels: {len(controller.weather_system.parcels)}",
            f"Cloud Ellipses: {len(cloud_ellipses)}",
            f"Cloud Cover: {result.get('cloud_cover', 0):.1f}%",
            f"Time: {result.get('time', 'N/A')}",
            "",
            "Controls:",
            "SPACE - Force spawn cloud",
            "ESC - Exit"
        ]
        
        for i, text in enumerate(debug_info):
            if text:  # Skip empty lines
                color = (255, 255, 0) if i < 5 else (255, 255, 255)
                surface = font.render(text, True, color, (0, 0, 0))
                screen.blit(surface, (10, 10 + i * 18))
        
        # Performance info
        fps_text = f"FPS: {clock.get_fps():.1f}"
        fps_surface = font.render(fps_text, True, (255, 255, 0), (0, 0, 0))
        screen.blit(fps_surface, (width - 100, 10))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
        frame_count += 1
        
        # Auto-spawn clouds periodically if none exist
        if frame_count % 120 == 0 and len(controller.weather_system.parcels) == 0:
            controller.weather_system._spawn(frame_count)
            print(f"Auto-spawn at frame {frame_count}")
    
    pygame.quit()
    print("Simulation ended")

if __name__ == "__main__":
    create_fixed_main()