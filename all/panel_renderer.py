import pygame
import numpy as np
import colorsys

def km_to_screen_coords(x_km, y_km, x_range, y_range, screen_width, screen_height):
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    screen_x = int((x_km - x_range[0]) / range_x * screen_width)
    screen_y = int((y_km - y_range[0]) / range_y * screen_height)
    return screen_x, screen_y

def draw_solar_panels(screen, panel_df, panel_coverage, power_output, x_range, y_range, width, height):
    """
    Draw solar panels with coverage and power output visualization.
    
    Returns:
        List of affected panel IDs sorted by coverage.
    """
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    panel_size_km = 0.4  # Default panel size
    panel_size_px = int(panel_size_km / range_x * width)
    
    affected_panels = []
    
    for _, row in panel_df.iterrows():
        panel_id = row["panel_id"]
        x_km = row["x_km"]
        y_km = row["y_km"]
        
        # Convert to screen coordinates
        x_px, y_px = km_to_screen_coords(x_km, y_km, x_range, y_range, width, height)
        
        # Determine panel color based on coverage and power
        coverage = panel_coverage.get(panel_id, 0.0)
        
        if coverage > 0:
            affected_panels.append((panel_id, coverage))
        
        # Get power output
        power_data = power_output.get(panel_id, {})
        power_value = power_data.get('final_power', 0)
        max_power = power_data.get('baseline', 1.0)
        
        # Normalize power for color
        power_pct = min(1.0, max(0.0, power_value / max_power if max_power > 0 else 0))
        
        # Calculate color from power percentage (blue to red)
        if power_pct > 0.8:
            # High power - blue to green gradient
            h = 0.6 - (power_pct - 0.8) * 0.6 / 0.2  # 0.6 (blue) to 0.3 (green)
            s = 0.8
            v = 0.9
        else:
            # Lower power - green to red gradient
            h = 0.3 - (0.8 - power_pct) * 0.3 / 0.8  # 0.3 (green) to 0 (red)
            s = 0.8
            v = 0.8
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        panel_color = (int(r * 255), int(g * 255), int(b * 255))
        
        # Draw panel
        panel_rect = pygame.Rect(
            x_px - panel_size_px//2, 
            y_px - panel_size_px//2, 
            panel_size_px, 
            panel_size_px
        )
        pygame.draw.rect(screen, panel_color, panel_rect)
        
        # Add border
        border_color = (0, 0, 0)
        pygame.draw.rect(screen, border_color, panel_rect, width=1)
        
        # Draw coverage indicator if covered
        if coverage > 0.05:
            # Draw shadow effect
            shadow_size = int(panel_size_px * coverage)
            shadow_color = (0, 0, 0, 120)  # Semi-transparent black
            
            shadow_surface = pygame.Surface((shadow_size, shadow_size), pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, shadow_color, (0, 0, shadow_size, shadow_size))
            
            shadow_pos = (
                x_px - shadow_size//2,
                y_px - shadow_size//2
            )
            screen.blit(shadow_surface, shadow_pos)
        
        # Draw panel ID if we don't have too many panels
        if len(panel_df) < 100:
            font_size = 9 if len(panel_df) < 60 else 7
            font = pygame.font.SysFont('Arial', font_size)
            label = font.render(panel_id, True, (255, 255, 255))
            label_rect = label.get_rect(center=(x_px, y_px))
            screen.blit(label, label_rect)
    
    # Sort affected panels by coverage
    affected_panels.sort(key=lambda x: x[1], reverse=True)
    return [panel_id for panel_id, _ in affected_panels]

def highlight_selected_panel(screen, panel_id, panel_df, x_range, y_range, width, height):
    """Highlight a selected panel with a bright border."""
    # Find the panel in the dataframe
    panel_row = panel_df[panel_df["panel_id"] == panel_id]
    if panel_row.empty:
        return
    
    x_km = panel_row.iloc[0]["x_km"]
    y_km = panel_row.iloc[0]["y_km"]
    
    # Convert to screen coordinates
    x_px, y_px = km_to_screen_coords(x_km, y_km, x_range, y_range, width, height)
    
    # Calculate panel size
    range_x = x_range[1] - x_range[0]
    panel_size_km = 0.4
    panel_size_px = int(panel_size_km / range_x * width)
    
    # Draw highlight
    highlight_rect = pygame.Rect(
        x_px - panel_size_px//2 - 3, 
        y_px - panel_size_px//2 - 3, 
        panel_size_px + 6, 
        panel_size_px + 6
    )
    
    # Pulsating effect based on time
    pulse = (pygame.time.get_ticks() % 1000) / 1000
    pulse_value = abs(pulse - 0.5) * 2  # 0 to 1 to 0
    
    highlight_color = (
        int(255 * pulse_value),
        int(255),
        int(100 + 155 * pulse_value)
    )
    
    pygame.draw.rect(screen, highlight_color, highlight_rect, width=2)
    
    # Draw info box with panel details
    info_box_width = 200
    info_box_height = 80
    info_box = pygame.Rect(
        min(width - info_box_width - 10, x_px + panel_size_px//2 + 10),
        y_px - info_box_height//2,
        info_box_width,
        info_box_height
    )
    
    # Info box background
    pygame.draw.rect(screen, (240, 240, 240, 220), info_box, border_radius=5)
    pygame.draw.rect(screen, (0, 0, 0), info_box, width=1, border_radius=5)
    
    # Info text
    font = pygame.font.SysFont('Arial', 12)
    title_font = pygame.font.SysFont('Arial', 14, bold=True)
    
    title = title_font.render(f"Panel {panel_id}", True, (0, 0, 0))
    screen.blit(title, (info_box.left + 10, info_box.top + 10))
    
    # Add position info
    pos_text = font.render(f"Position: ({x_km:.1f}, {y_km:.1f}) km", True, (0, 0, 0))
    screen.blit(pos_text, (info_box.left + 10, info_box.top + 30))
    
    # Add more panel details if available
    # (This would be extended with power data, etc.)