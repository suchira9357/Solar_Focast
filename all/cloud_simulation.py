import math
import random
import numpy as np
import sim_config as CFG
from enhanced_wind_field import EnhancedWindField

# --- Cloud Type Appearance Profiles ------------------------
CLOUD_TYPE_PROFILES = {
    "cirrus": {
        "count": 1,
        "cw_range": (3000, 5000),
        "ch_range": (400, 800),
        "opacity_range": (0.2, 0.4),
        "rotation_range": (-0.4, 0.4)
    },
    "cumulus": {
        "count": (3, 5),
        "cw_range": (1200, 2000),
        "ch_range": (1200, 2000),
        "opacity_range": (0.5, 0.7),
        "rotation_range": (-0.1, 0.1)
    },
    "cumulonimbus": {
        "count": (8, 12),
        "cw_range": (1500, 2500),
        "ch_range": (2500, 4500),
        "opacity_range": (0.8, 1.0),
        "rotation_range": (-0.05, 0.05)
    }
}

def _altitude_to_index(alt_km, layer_heights):
    """
    Map altitude in km to a layer index.
    
    Args:
        alt_km: Altitude in kilometers
        layer_heights: List of layer boundary heights in km
        
    Returns:
        Layer index (0 for lowest layer, N-2 for highest layer)
    """
    for i in range(len(layer_heights) - 1):
        if alt_km >= layer_heights[i] and alt_km < layer_heights[i+1]:
            return i
    return len(layer_heights) - 2  # Return the highest layer index

def generate_ellipses_for_type(cloud_type, center_x, center_y):
    profile = CLOUD_TYPE_PROFILES[cloud_type]
    count = profile["count"]
    if isinstance(count, tuple):
        puff_count = random.randint(*count)
    else:
        puff_count = count
    
    ellipses = []
    
    # Handle specific cloud types differently
    if cloud_type == "cirrus":
        # Single long, thin, rotated ellipse
        cw = random.randint(*profile["cw_range"])
        ch = random.randint(*profile["ch_range"])
        crot = random.uniform(*profile["rotation_range"])
        cop = random.uniform(*profile["opacity_range"])
        ellipses.append((center_x, center_y, cw, ch, crot, cop))
    
    elif cloud_type == "cumulus":
        # 3-5 fluffy, round clustered ellipses
        for _ in range(puff_count):
            # Random offsets for clustering
            dx = random.uniform(-500, 500)
            dy = random.uniform(-500, 500)
            cx = center_x + dx
            cy = center_y + dy
            cw = random.randint(*profile["cw_range"])
            ch = random.randint(*profile["ch_range"])
            crot = random.uniform(*profile["rotation_range"])
            cop = random.uniform(*profile["opacity_range"])
            ellipses.append((cx, cy, cw, ch, crot, cop))
    
    elif cloud_type == "cumulonimbus":
        # 8-12 vertically extended ellipses
        y_offsets = np.linspace(-2000, 2000, puff_count)
        for i, y_offset in enumerate(y_offsets):
            # Horizontal position varies less at top (anvil shape)
            x_range = 600 if i < puff_count//2 else 300
            dx = random.uniform(-x_range, x_range)
            cx = center_x + dx
            cy = center_y + y_offset
            
            # Width increases at top for anvil effect
            width_factor = 1.0 if i < puff_count//2 else 1.5
            cw = random.randint(*profile["cw_range"]) * width_factor
            
            # Height decreases at top
            height_factor = 1.0 if i < puff_count//2 else 0.7
            ch = random.randint(*profile["ch_range"]) * height_factor
            
            crot = random.uniform(*profile["rotation_range"])
            
            # Opacity varies - darker at bottom, lighter at top
            opacity_factor = 1.0 if i < puff_count//2 else 0.8
            cop = random.uniform(*profile["opacity_range"]) * opacity_factor
            
            ellipses.append((cx, cy, cw, ch, crot, cop))
    
    return ellipses

def generate_shifted_ellipses(source_puffs, dx, dy, duration):
    shifted = []
    for puff in source_puffs:
        cx, cy, cw, ch, crot, cop = puff
        new_cx = cx + dx * duration * 1000  # Convert km to meters
        new_cy = cy + dy * duration * 1000
        shifted.append((new_cx, new_cy, cw, ch, crot, cop))
    return shifted

def interpolate_ellipses(src, tgt, t):
    return [
        (
            src[i][0] * (1 - t) + tgt[i][0] * t,
            src[i][1] * (1 - t) + tgt[i][1] * t,
            src[i][2] * (1 - t) + tgt[i][2] * t,
            src[i][3] * (1 - t) + tgt[i][3] * t,
            src[i][4] * (1 - t) + tgt[i][4] * t,
            src[i][5] * (1 - t) + tgt[i][5] * t,
        )
        for i in range(len(src))
    ]

M_PER_FRAME = CFG.PHYSICS_TIMESTEP

class CloudParcel:
    def __init__(self, x, y, wind, ctype):
        self.x, self.y = x, y
        self.type = ctype
        preset = CFG.CLOUD_TYPES[ctype]
        self.alt = preset["alt_km"]
        r_lo, r_hi = preset["r_km"]
        self.r = random.uniform(r_lo, r_hi)
        self.max_op = preset["opacity_max"]
        self.opacity = 0.0
        self.vx = self.vy = 0.0
        self.wind = wind
        self.position_history_x = []
        self.position_history_y = []
        self.speed_k = preset["speed_k"]

    def _update_velocity(self, t):
        s, h = self.wind.sample(self.x, self.y, self.alt * 1000)
        s *= self.speed_k
        vx_new = s * M_PER_FRAME * math.cos(math.radians(h))
        vy_new = s * M_PER_FRAME * math.sin(math.radians(h))
        self.vx = 0.8 * vx_new + 0.2 * self.vx
        self.vy = 0.8 * vy_new + 0.2 * self.vy

    def update(self, t, hum):
        # For backward compatibility - now calls step with default parameters
        return self.step(CFG.PHYSICS_TIMESTEP, self.wind)

    def step(self, timestep_sec, wind_field=None, sim_time=None):
        """
        Update cloud parcel position based on wind velocity.
        
        Args:
            timestep_sec: Time step in seconds
            wind_field: Wind field object providing velocity
            sim_time: Optional simulation time
            
        Returns:
            Boolean indicating if the cloud should be removed
        """
        # Store position history
        if len(self.position_history_x) >= CFG.POSITION_HISTORY_LENGTH:
            self.position_history_x.pop(0)
            self.position_history_y.pop(0)
        self.position_history_x.append(self.x)
        self.position_history_y.append(self.y)
        
        # Get velocity from wind field using altitude-based vector
        if sim_time is not None and hasattr(wind_field, 'vector_at_altitude'):
            # Convert altitude to index
            alt_idx = _altitude_to_index(self.alt, CFG.LAYER_HEIGHTS)
            # Get velocity vector
            dx_km_min, dy_km_min = wind_field.vector_at_altitude(sim_time / 60.0, alt_idx)
            # Convert km/min to m/s
            vx_ms = dx_km_min * 1000 / 60
            vy_ms = dy_km_min * 1000 / 60
        else:
            # Use sample method
            s, h = wind_field.sample(self.x, self.y, self.alt * 1000)
            s *= self.speed_k
            # Convert to vector components
            dir_rad = math.radians(h)
            vx_ms = s * math.cos(dir_rad)
            vy_ms = s * math.sin(dir_rad)
        
        # Update velocity with smoothing
        vx_new = vx_ms * M_PER_FRAME
        vy_new = vy_ms * M_PER_FRAME
        self.vx = CFG.WIND_SMOOTH * vx_new + (1 - CFG.WIND_SMOOTH) * self.vx
        self.vy = CFG.WIND_SMOOTH * vy_new + (1 - CFG.WIND_SMOOTH) * self.vy
        
        # Update position
        self.x += self.vx * CFG.MOVEMENT_MULTIPLIER
        self.y += self.vy * CFG.MOVEMENT_MULTIPLIER
        
        # Handle domain wrapping
        d = CFG.DOMAIN_SIZE_M
        b = CFG.SPAWN_BUFFER_M
        if self.x < -b: self.x = d + b
        elif self.x > d + b: self.x = -b
        if self.y < -b: self.y = d + b
        elif self.y > d + b: self.y = -b
        
        # Update cloud size
        target_size = (CFG.CLOUD_TYPES[self.type]["r_km"][0] + CFG.CLOUD_TYPES[self.type]["r_km"][1]) / 2
        growth_rate = (target_size - self.r) * 0.01 + random.uniform(-0.01, 0.01)
        self.r = np.clip(self.r + growth_rate, 0.2, CFG.R_MAX_KM if hasattr(CFG, 'R_MAX_KM') else 6.0)
        
        # Update opacity
        size_factor = min(1.0, self.r / (CFG.CLOUD_TYPES[self.type]["r_km"][1] * 0.8))
        self.opacity = size_factor * self.max_op
        
        # Return False to keep cloud, True to remove it
        return self.r < 0.25 or (random.random() < 0.001)

    def ellipse(self):
        diam = self.r * 2000
        return (self.x, self.y, diam, diam, 0, self.opacity, self.alt, self.type)

class EnhancedCloudParcel(CloudParcel):
    def __init__(self, x, y, wind, ctype):
        super().__init__(x, y, wind, ctype)
        self.cloud_type = ctype  # Store cloud type
        self.spawn_x = x  # Track original spawn location
        self.spawn_y = y
        
        # Generate initial puffs
        self.source_puffs = generate_ellipses_for_type(self.cloud_type, self.x, self.y)
        
        # Calculate initial velocity and direction for interpolation
        self.t_duration = 10.0  # seconds to cross
        
        # Initialize with same target as source to prevent visual jumps
        self.target_puffs = self.source_puffs[:]
        
        # Initialize interpolation timer
        self.t = 0.0
    
    def step(self, timestep_sec, wind_field=None, sim_time=None):
        """
        Update cloud parcel position based on wind velocity.
        
        Args:
            timestep_sec: Time step in seconds
            wind_field: Optional wind field for custom vectors
            sim_time: Optional simulation time
            
        Returns:
            Boolean indicating if the cloud should be removed
        """
        # Store position for history
        if len(self.position_history_x) >= CFG.POSITION_HISTORY_LENGTH:
            self.position_history_x.pop(0)
            self.position_history_y.pop(0)
        self.position_history_x.append(self.x)
        self.position_history_y.append(self.y)
        
        # Get velocity from wind field using altitude-based vector
        if sim_time is not None and hasattr(wind_field, 'vector_at_altitude'):
            # Convert altitude to index
            alt_idx = _altitude_to_index(self.alt, CFG.LAYER_HEIGHTS)
            # Get velocity vector (km/min)
            vx_kmm, vy_kmm = wind_field.vector_at_altitude(sim_time / 60.0, alt_idx)
            # Convert km/min to m/s
            vx_ms = vx_kmm * 1000 / 60
            vy_ms = vy_kmm * 1000 / 60
        else:
            # Use sample method if time not provided
            speed, direction = wind_field.sample(self.x, self.y, self.alt * 1000)
            speed *= self.speed_k  # Apply cloud type speed factor
            # Convert to vector components
            direction_rad = math.radians(direction)
            vx_ms = speed * math.cos(direction_rad)
            vy_ms = speed * math.sin(direction_rad)
        
        # Update velocity with smoothing
        vx_new = vx_ms * M_PER_FRAME
        vy_new = vy_ms * M_PER_FRAME
        self.vx = CFG.WIND_SMOOTH * vx_new + (1 - CFG.WIND_SMOOTH) * self.vx
        self.vy = CFG.WIND_SMOOTH * vy_new + (1 - CFG.WIND_SMOOTH) * self.vy
        
        # Update position
        self.x += self.vx * CFG.MOVEMENT_MULTIPLIER
        self.y += self.vy * CFG.MOVEMENT_MULTIPLIER
        
        # Handle domain wrapping
        d = CFG.DOMAIN_SIZE_M
        b = CFG.SPAWN_BUFFER_M
        if self.x < -b: self.x = d + b
        elif self.x > d + b: self.x = -b
        if self.y < -b: self.y = d + b
        elif self.y > d + b: self.y = -b
        
        # Update cloud shape interpolation
        self.t += timestep_sec / self.t_duration
        if self.t >= 1.0:
            # Reset interpolation
            self.t = 0.0
            
            # Current puffs become source puffs
            self.source_puffs = self.get_current_puffs()
            
            # Generate new target puffs based on current position
            self.target_puffs = generate_ellipses_for_type(self.cloud_type, self.x, self.y)
        
        # Update cloud size and opacity (from parent class)
        # Calculate cloud size
        target_size = (CFG.CLOUD_TYPES[self.type]["r_km"][0] + CFG.CLOUD_TYPES[self.type]["r_km"][1]) / 2
        growth_rate = (target_size - self.r) * 0.01 + random.uniform(-0.01, 0.01)
        self.r = np.clip(self.r + growth_rate, 0.2, CFG.R_MAX_KM if hasattr(CFG, 'R_MAX_KM') else 6.0)
        
        # Update opacity
        size_factor = min(1.0, self.r / (CFG.CLOUD_TYPES[self.type]["r_km"][1] * 0.8))
        self.opacity = size_factor * self.max_op
        
        # Return False to keep cloud, True to remove it
        return self.r < 0.25 or (random.random() < 0.001)
    
    def get_current_puffs(self):
        """Return interpolated puffs adjusted by parcel motion"""
        # Interpolate between source and target puffs (t ∈ [0,1])
        puffs = interpolate_ellipses(self.source_puffs, self.target_puffs, self.t)
        
        # Determine movement offset from original puff center to current parcel position
        dx = self.x - self.spawn_x
        dy = self.y - self.spawn_y
        
        # Apply offset to every puff
        adjusted = []
        for puff in puffs:
            cx, cy, cw, ch, crot, cop = puff
            adjusted.append((cx + dx, cy + dy, cw, ch, crot, cop))
        
        return adjusted
    
    def get_ellipses(self):
        """Return list of ellipses with altitude and type appended"""
        result = []
        current_puffs = self.get_current_puffs()
        for e in current_puffs:
            cx, cy, cw, ch, crot, cop = e
            # Append altitude and cloud type to each ellipse
            result.append((cx, cy, cw, ch, crot, cop, self.alt, self.cloud_type))
        return result

class WeatherSystem:
    def __init__(self, seed=0):
        self.wind = EnhancedWindField()
        self.parcels = []
        self.sim_time = 0.0

    def _spawn(self, t):
        """
        Spawn a new cloud parcel at the NW region of the domain.
        
        Args:
            t: Current simulation time
        """
        # Get wind direction from wind field to determine spawn location
        _, hdg = self.wind.sample(CFG.DOMAIN_SIZE_M/2, CFG.DOMAIN_SIZE_M/2, 1000)
        d = CFG.DOMAIN_SIZE_M; b = CFG.SPAWN_BUFFER_M
        
        # Spawn in NW region (for NW to SE motion)
        x = random.uniform(0.05 * CFG.DOMAIN_SIZE_M, 0.2 * CFG.DOMAIN_SIZE_M)
        y = random.uniform(0.05 * CFG.DOMAIN_SIZE_M, 0.2 * CFG.DOMAIN_SIZE_M)
        
        # Select cloud type based on weights
        ctype = random.choices(list(CFG.CLOUD_TYPE_WEIGHTS.keys()),
                              weights=list(CFG.CLOUD_TYPE_WEIGHTS.values()))[0]
        
        # Create new cloud parcel with enhanced visualization
        self.parcels.append(EnhancedCloudParcel(x, y, self.wind, ctype))

    def step(self, t=None, dt=None, t_s=None):
        """
        Update the weather system for one time step.
        
        Args:
            t: Frame count or time value
            dt: Time step in seconds
            t_s: Simulation time in seconds (optional)
        """
        # Update simulation time
        if t_s is not None:
            self.sim_time = t_s
        elif t is not None:
            self.sim_time = t
        else:
            self.sim_time += 1
        
        # Use physics timestep if dt not provided
        if dt is None:
            dt = CFG.PHYSICS_TIMESTEP
        
        # Update wind field
        self.wind.step(self.sim_time)
        
        # Update all parcels with wind
        expired_indices = []
        for i, parcel in enumerate(self.parcels):
            if parcel.step(dt, self.wind, self.sim_time):
                expired_indices.append(i)
        
        # Remove expired parcels (in reverse to preserve indices)
        for i in sorted(expired_indices, reverse=True):
            if i < len(self.parcels):
                self.parcels.pop(i)
        
        # Spawn new clouds if needed
        while len(self.parcels) < CFG.MAX_PARCELS:
            self._spawn(self.sim_time)

    def get_avg_trajectory(self):
        if not self.parcels:
            return None, None, 0
        speeds = []
        directions_x = []
        directions_y = []
        for parcel in self.parcels:
            speed = math.sqrt(parcel.vx**2 + parcel.vy**2)
            direction = math.degrees(math.atan2(parcel.vy, parcel.vx)) % 360
            speeds.append(speed)
            direction_rad = math.radians(direction)
            directions_x.append(math.cos(direction_rad))
            directions_y.append(math.sin(direction_rad))
        avg_speed = sum(speeds) / len(speeds)
        avg_dir_x = sum(directions_x) / len(directions_x)
        avg_dir_y = sum(directions_y) / len(directions_y)
        avg_direction = math.degrees(math.atan2(avg_dir_y, avg_dir_x)) % 360
        mean_vector_length = math.sqrt(avg_dir_x**2 + avg_dir_y**2)
        confidence = mean_vector_length
        avg_speed_kmh = avg_speed * 3.6 * CFG.MOVEMENT_MULTIPLIER  # Convert m/s to km/h
        return avg_speed_kmh, avg_direction, confidence
        
    def current_cloud_cover_pct(self):
        total_area = 0
        domain_area = CFG.AREA_SIZE_KM * CFG.AREA_SIZE_KM
        for p in self.parcels:
            cloud_area = math.pi * (p.r ** 2)
            total_area += cloud_area
        cover = min(100, (total_area / domain_area) * 100 * 5)
        return cover

def collect_visible_ellipses(parcels):
    """Collect all visible ellipses from all parcels"""
    ellipses = []
    for parcel in parcels:
        if isinstance(parcel, EnhancedCloudParcel):
            ellipses.extend(parcel.get_ellipses())
        else:
            ellipses.append(parcel.ellipse())
    return ellipses