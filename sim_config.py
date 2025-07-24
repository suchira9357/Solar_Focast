# sim_config.py
# Add your simulation configuration variables here

# Example placeholder variables:
SIMULATION_NAME = "Default Simulation"
SIMULATION_VERSION = "1.0"

# Simulation configuration variables required by main.py
FPS = 60
AREA_SIZE_KM = 50
DOMAIN_SIZE_M = 10000
BASE_WIND_SPEED = 2.0
MOVEMENT_MULTIPLIER = 1.0
SINGLE_CLOUD_MODE = True
SPAWN_PROBABILITY = 1.0
# Wind grid configuration required by enhanced_wind_field.py
WIND_GRID = 100  # Default grid resolution, adjust as needed

# Wind direction configuration required by enhanced_wind_field.py
BASE_WIND_DIRECTION = 0.0  # Default wind direction in degrees (0 = east, 90 = north)

# Cloud transmittance configuration required by shadow_calculator
CLOUD_TRANSMITTANCE = 0.5  # Default value, adjust as needed (0 = fully opaque, 1 = fully transparent)

# Shadow fade configuration required by shadow_calculator
SHADOW_FADE_MS = 500  # Default fade duration in milliseconds

# Add more configuration variables as needed
