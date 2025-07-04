# cloud_renderer.py
"""
Cloud Renderer for Solar Farm Simulation
Handles visualization of different cloud types using Pygame
"""
import pygame
import math
import numpy as np
import sim_config as CFG
import random


def create_cloud_surface(ellipse, dom_m, area_km, w_px, h_px, x_rng, y_rng):
    """
    Create a pygame surface for a cloud.
    
    Args:
        ellipse: Tuple of (x, y, width, height, rotation, opacity, altitude, type)
        dom_m: Domain size in meters
        area_km: Area size in kilometers
        w_px, h_px: Screen dimensions in pixels
        x_rng, y_rng: Coordinate ranges in km
        
    Returns:
        Tuple of (surface, position) for blitting to screen
    """
    # Extract cloud parameters
    x_m, y_m, w_m, h_m, rot, alpha, *_rest = ellipse
    ctype = _rest[-1] if _rest else "cumulus"  # Default to cumulus if type not specified
    
    # Calculate scale factors from domain coordinates to screen pixels
    scale_x = w_px / (x_rng[1]-x_rng[0]) / 1000
    scale_y = h_px / (y_rng[1]-y_rng[0]) / 1000
    
    # Calculate dimensions in pixels
    diam_x = int(w_m * scale_x)
    diam_y = int(h_m * scale_y)
    
    # Ensure minimum size
    diam_x = max(5, diam_x)
    diam_y = max(5, diam_y)
    
    # Create surface with alpha channel
    surf = pygame.Surface((diam_x, diam_y), pygame.SRCALPHA)
    
    # Render based on cloud type
    if ctype == "cirrus":
        # Cirrus: thin, light-grey, flattened cloud
        color = (230, 230, 230, int(alpha*255))  # Light grey
        pygame.draw.ellipse(surf, color, (0, 0, diam_x, diam_y))
        
        # Add slight texture
        for _ in range(3):
            x_offset = random.randint(0, diam_x//4)
            y_offset = random.randint(0, diam_y//3)
            width = diam_x - 2 * x_offset
            height = max(1, diam_y - 2 * y_offset)
            highlight = (245, 245, 245, int(alpha*255*0.7))
            pygame.draw.ellipse(surf, highlight, (x_offset, y_offset, width, height))
    
    elif ctype == "cumulonimbus":
        # Cumulonimbus: dark base + light anvil
        # Draw dark base
        base_col = (180, 180, 180, int(alpha*255*0.9))  # Darker grey
        pygame.draw.ellipse(surf, base_col, (0, 0, diam_x, diam_y))
        
        # Add slight shading for 3D effect
        if diam_x > 20 and diam_y > 20:
            shadow_col = (160, 160, 160, int(alpha*255*0.5))
            pygame.draw.ellipse(surf, shadow_col, (diam_x//8, diam_y//8, diam_x*3//4, diam_y*3//4))
    
    else:  # Default: cumulus
        # Cumulus: white puffy cloud
        color = (255, 255, 255, int(alpha*255))  # White
        pygame.draw.ellipse(surf, color, (0, 0, diam_x, diam_y))
        
        # Add slight highlight for 3D effect if large enough
        if diam_x > 15 and diam_y > 15:
            highlight = (255, 255, 255, int(alpha*255*0.8))
            pygame.draw.ellipse(surf, highlight, (diam_x//6, diam_y//6, diam_x*2//3, diam_y*2//3))
    
    # Calculate position for rendering (centered on cloud position)
    pos = (int(x_m*scale_x - diam_x/2), int(y_m*scale_y - diam_y/2))
    
    return surf, pos

def draw_cloud_trail(screen, positions, x_range, y_range, width, height):
    """
    Draw a trail showing cloud movement history.
    
    Args:
        screen: Pygame surface to draw on
        positions: List of (x_km, y_km) cloud positions
        x_range, y_range: Coordinate ranges in km
        width, height: Screen dimensions in pixels
    """
    if len(positions) < 2:
        return
    
    # Convert km positions to screen coordinates
    screen_positions = []
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    
    for x_km, y_km in positions:
        screen_x = int((x_km - x_range[0]) / range_x * width)
        screen_y = int((y_km - y_range[0]) / range_y * height)
        screen_positions.append((screen_x, screen_y))
    
    # Draw lines connecting positions
    for i in range(1, len(screen_positions)):
        start_pos = screen_positions[i-1]
        end_pos = screen_positions[i]
        
        # Increasing alpha for more recent positions
        alpha = int(i / len(screen_positions) * 180)
        color = (100, 100, 250, alpha)
        
        pygame.draw.line(screen, color, start_pos, end_pos, width=2)