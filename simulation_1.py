from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os  # Import the 'os' module for file operations
import csv # Import the 'csv' module for CSV file handling

# --- Configuration ---
area_size_km = 10.0
image_pixels = 500
num_solar_panels = 20 # Updated to 20 as we have 20 locations
num_clouds = 20
solar_panel_size = 20
panel_label_font_size = 6
axis_label_font_size = 8
axis_color = (0, 0, 0)
panel_label_color = (255, 255, 255)

# --- Coordinate System Configuration ---
x_axis_start_pixel = 50
y_axis_start_pixel = 50
axis_length_pixels = image_pixels - 100
x_axis_end_pixel = x_axis_start_pixel + axis_length_pixels
y_axis_end_pixel = y_axis_start_pixel + axis_length_pixels
x_km_start = 0.0
x_km_end = area_size_km
y_km_start = 0.0
y_km_end = area_size_km
x_ticks_km = 1.0
y_ticks_km = 1.0

# --- File path for storing panel locations (Not used in this manual location version, but kept for consistency) ---
PANEL_LOCATIONS_FILE = "solar_panel_locations.csv"

# --- Step 1: Create Background with Coordinate Axes ---
background_color = (102, 204, 102)
background_image = Image.new("RGB", (image_pixels, image_pixels), background_color)
draw = ImageDraw.Draw(background_image)

# Load fonts
try:
    axis_label_font = ImageFont.truetype("arial.ttf", axis_label_font_size)
    panel_label_font = ImageFont.truetype("arial.ttf", panel_label_font_size)
except IOError:
    axis_label_font = ImageFont.load_default()
    panel_label_font = ImageFont.load_default()

# Draw X and Y axes (same as before)
draw.line([(x_axis_start_pixel, y_axis_end_pixel), (x_axis_end_pixel, y_axis_end_pixel)], fill=axis_color, width=1) # X-axis
draw.line([(x_axis_start_pixel, y_axis_start_pixel), (x_axis_start_pixel, y_axis_end_pixel)], fill=axis_color, width=1) # Y-axis

# Add X-axis tick marks and labels (same as before)
num_x_ticks = int((x_km_end - x_km_start) / x_ticks_km) + 1
for i in range(num_x_ticks):
    km_value = x_km_start + i * x_ticks_km
    pixel_x = x_axis_start_pixel + int((km_value / area_size_km) * axis_length_pixels)
    draw.line([(pixel_x, y_axis_end_pixel), (pixel_x, y_axis_end_pixel + 5)], fill=axis_color, width=1) # Tick mark
    draw.text((pixel_x, y_axis_end_pixel + 7), f"{km_value:.0f}", fill=axis_color, font=axis_label_font, anchor="mt") # Label

# Add Y-axis tick marks and labels (same as before)
num_y_ticks = int((y_km_end - y_km_start) / y_ticks_km) + 1
for i in range(num_y_ticks):
    km_value = y_km_start + i * y_ticks_km
    pixel_y = y_axis_end_pixel - int((km_value / area_size_km) * axis_length_pixels) # Y-axis is inverted in pixels
    draw.line([(x_axis_start_pixel, pixel_y), (x_axis_start_pixel - 5, pixel_y)], fill=axis_color, width=1) # Tick mark
    draw.text((x_axis_start_pixel - 7, pixel_y), f"{km_value:.0f}", fill=axis_color, font=axis_label_font, anchor="rm") # Label

# --- Step 2: Place Solar Panels with PREDEFINED X,Y Coordinate Labels ---
solar_panel_color = (50, 50, 150)
panel_locations_km = []
panel_ids = {}
panel_rectangles = [] # List to store panel rectangles for overlap check (though overlaps are less likely with fixed locations)

# --- Predefined panel locations in KM (X, Y) ---
manual_panel_locations_km = [
    (8.3, 4.2),
    (1.6, 9.8),
    (6.7, 5.1),
    (9.2, 0.8),
    (4.5, 7.1),
    (7.3, 8.6),
    (0.6, 6.3),
    (3.9, 2.0),
    (5.8, 8.4),
    (3.1, 4.9),
    (8.2, 9.9),
    (5.7, 5.6),
    (2.9, 0.3),
    (6.4, 7.8),
    (9.3, 3.9),
    (5.1, 6.9),
    (7.0, 9.2),
    (1.2, 4.6),
    (9.5, 2.5),
    (5.1, 1.2),
]
assert len(manual_panel_locations_km) == num_solar_panels, "Number of manual locations must match num_solar_panels"


def check_overlap(new_panel_rect, existing_panel_rectangles):
    """Checks if new_panel_rect overlaps with any rectangles in existing_panel_rectangles."""
    for existing_rect in existing_panel_rectangles:
        # Check for overlap in X dimension AND Y dimension
        if (new_panel_rect[0][0] < existing_rect[1][0] and new_panel_rect[1][0] > existing_rect[0][0] and
            new_panel_rect[0][1] < existing_rect[1][1] and new_panel_rect[1][1] > existing_rect[0][1]):
            return True  # Overlap detected
    return False  # No overlap

panel_id_counter = 1 # Initialize panel ID counter

# Iterate through the manually defined locations
for location_km in manual_panel_locations_km:
    x_km, y_km = location_km

    panel_locations_km.append((x_km, y_km)) # Store locations

    # Convert km coordinates to pixel coordinates
    x_pixel = x_axis_start_pixel + int((x_km / area_size_km) * axis_length_pixels)
    y_pixel = y_axis_end_pixel - int((y_km / area_size_km) * axis_length_pixels) # Invert Y for pixel coords

    # Calculate pixel rectangle for the new panel
    new_panel_rect = [
        (x_pixel - solar_panel_size // 2, y_pixel - solar_panel_size // 2),
        (x_pixel + solar_panel_size // 2, y_pixel + solar_panel_size // 2)
    ]

    # Overlap check (optional for manual locations, but kept for robustness)
    if not check_overlap(new_panel_rect, panel_rectangles):
        panel_rectangles.append(new_panel_rect) # Add rectangle to the list
        panel_ids[f"P{panel_id_counter}"] = (x_km, y_km)


        # Draw solar panel
        draw.rectangle(new_panel_rect, fill=solar_panel_color)

        # Draw coordinate label (X,Y in km) above the panel
        label_x = x_pixel
        label_y = y_pixel - solar_panel_size // 2 - 5  # Position label slightly above panel
        label_text = f"({x_km:.1f},{y_km:.1f})" # Format coordinates to 1 decimal place
        draw.text((label_x, label_y), label_text, fill=panel_label_color, font=panel_label_font, anchor="mb") # "mb" anchor: middle-bottom
        panel_id_counter += 1
    else:
        print(f"Warning: Overlap detected at location (X={x_km:.2f}, Y={y_km:.2f}) - Panel not placed.") # Indicate if overlap occurs with manual locations


# --- Step 3 & 4: Generate & Place Clouds (Simplified Clouds - Ellipses) ---
cloud_color = (200, 200, 200, 180)

for _ in range(num_clouds):
    cloud_width = int(np.random.rand() * 100 + 10)
    cloud_height = int(np.random.rand() * 50 + 30)
    cloud_x = int(np.random.rand() * image_pixels - cloud_width // 2)
    cloud_y = int(np.random.rand() * image_pixels - cloud_height // 2)

    cloud_box = [(cloud_x, cloud_y), (cloud_x + cloud_width, cloud_y + cloud_height)]
    draw.ellipse(cloud_box, fill=cloud_color)


# --- Step 5: Visualization ---
plt.imshow(background_image)
plt.title("Solar Panel Simulation with X,Y Coordinates (km)")
plt.axis('off')
plt.show()

# --- Optional: Save the image ---
# background_image.save("solar_simulation_xy_coords_manual_locations.png")

# Print Panel Locations and IDs
for panel_id, location_km in panel_ids.items():
    print(f"Panel ID: {panel_id}, Location (km): X={location_km[0]:.2f}, Y={location_km[1]:.2f}")