"""
Optimized Background module for solar farm simulation
Handles panel layout, background grid, and static elements with vectorized operations
"""
import os
import numpy as np
import pandas as pd
import pygame
import math
import colorsys

# OpenGL imports - only imported if OpenGL rendering is used
try:
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
    import ctypes
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available in background_module. Using Pygame renderer only.")

def create_panel_dataframe(coordinates_path=None, csv_file=None, num_panels=None):
    """
    Create a DataFrame with solar panel information.
    
    Can load from:
    - coordinates_path/csv_file: Path to CSV with panel coordinates
    - num_panels: Number of panels to generate in a grid pattern
    
    Returns DataFrame with:
    - panel_id: Unique identifier
    - x_km, y_km: Position in kilometers
    - power_capacity: Maximum power in kW
    """
    if coordinates_path is not None or csv_file is not None:
        # Load from file
        path = coordinates_path if coordinates_path is not None else csv_file
        try:
            df = pd.read_csv(path)
            # Ensure required columns exist
            if 'panel_id' not in df.columns:
                df['panel_id'] = [f"P{i+1:03d}" for i in range(len(df))]
            if 'power_capacity' not in df.columns:
                df['power_capacity'] = 5.0  # Default capacity in kW
                
            print(f"Loaded {len(df)} panels from {path}")
            return df
        except Exception as e:
            print(f"Error loading panel coordinates from {path}: {e}")
            print("Falling back to generated panels")
    
    # Generate panels in a grid pattern
    if num_panels is None:
        num_panels = 36  # Default to 6x6 grid
    
    # Determine grid dimensions (approximate square)
    grid_size = int(np.ceil(np.sqrt(num_panels)))
    
    # Generate coordinates vectorized
    spacing = 3.0  # km between panels
    margin = 5.0   # km from edges
    
    # Create grid indices
    indices = np.arange(num_panels)
    i_coords = indices % grid_size
    j_coords = indices // grid_size
    
    # Vectorized coordinate calculation
    x_coords = margin + i_coords * spacing + np.random.uniform(-0.5, 0.5, num_panels)
    y_coords = margin + j_coords * spacing + np.random.uniform(-0.5, 0.5, num_panels)
    
    # Create DataFrame directly with vectorized data
    df = pd.DataFrame({
        'panel_id': [f"P{i+1:03d}" for i in range(num_panels)],
        'x_km': x_coords,
        'y_km': y_coords,
        'power_capacity': np.random.uniform(4.0, 6.0, num_panels)
    })
    
    print(f"Generated {len(df)} panels in a grid pattern")
    return df

# Pre-compute common grid spacings to avoid repeated calculations
_GRID_SPACING_CACHE = {}

def _get_grid_spacing(range_size):
    """Get cached grid spacing for given range size"""
    if range_size not in _GRID_SPACING_CACHE:
        _GRID_SPACING_CACHE[range_size] = 5.0 if range_size > 20 else 2.0
    return _GRID_SPACING_CACHE[range_size]

def setup_background(screen, width, height, x_range, y_range, grid_color=(204, 204, 204)):
    """
    Optimized background grid drawing with vectorized operations.
    
    Args:
        screen: Pygame surface to draw on
        width, height: Screen dimensions
        x_range, y_range: Coordinate ranges in km
        grid_color: Color for grid lines
    """
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    
    # Get grid spacing (cached)
    grid_interval_km = _get_grid_spacing(range_x)
    
    # Vectorized grid line calculations
    x_km_lines = np.arange(x_range[0], x_range[1] + grid_interval_km, grid_interval_km)
    y_km_lines = np.arange(y_range[0], y_range[1] + grid_interval_km, grid_interval_km)
    
    # Vectorized coordinate conversion
    x_px_lines = ((x_km_lines - x_range[0]) / range_x * width).astype(int)
    y_px_lines = ((y_km_lines - y_range[0]) / range_y * height).astype(int)
    
    # Batch draw vertical lines
    for x_px in x_px_lines:
        pygame.draw.line(screen, grid_color, (x_px, 0), (x_px, height), 1)
    
    # Batch draw horizontal lines
    for y_px in y_px_lines:
        pygame.draw.line(screen, grid_color, (0, y_px), (width, y_px), 1)
    
    # Optimized label drawing with reduced frequency
    font = pygame.font.SysFont('Arial', 12)
    tick_interval = 10 if range_x > 20 else 5
    
    # X-axis labels (vectorized selection)
    x_label_values = np.arange(int(x_range[0]), int(x_range[1])+1, tick_interval)
    x_label_positions = ((x_label_values - x_range[0]) / range_x * width).astype(int)
    
    for x_val, x_px in zip(x_label_values, x_label_positions):
        label = font.render(f"{int(x_val)}", True, (0, 0, 0))
        screen.blit(label, (x_px - 5, height - 20))
    
    # Y-axis labels (vectorized selection)
    y_label_values = np.arange(int(y_range[0]), int(y_range[1])+1, tick_interval)
    y_label_positions = ((y_label_values - y_range[0]) / range_y * height).astype(int)
    
    for y_val, y_px in zip(y_label_values, y_label_positions):
        label = font.render(f"{int(y_val)}", True, (0, 0, 0))
        screen.blit(label, (5, y_px - 10))
    
    # Axis titles (only draw if not cached)
    x_label = font.render("Distance (km)", True, (0, 0, 0))
    screen.blit(x_label, (width // 2 - 40, height - 20))
    
    y_label = font.render("Distance (km)", True, (0, 0, 0))
    y_label = pygame.transform.rotate(y_label, 90)
    screen.blit(y_label, (5, height // 2 - 40))

def km_to_screen_coords_vectorized(x_km_array, y_km_array, x_range, y_range, screen_width, screen_height):
    """Vectorized version of km_to_screen_coords for multiple points"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    
    screen_x = ((x_km_array - x_range[0]) / range_x * screen_width).astype(int)
    screen_y = ((y_km_array - y_range[0]) / range_y * screen_height).astype(int)
    
    return screen_x, screen_y

def km_to_screen_coords(x_km, y_km, x_range, y_range, screen_width, screen_height):
    """Convert km coordinates to screen pixels (single point)"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    screen_x = int((x_km - x_range[0]) / range_x * screen_width)
    screen_y = int((y_km - y_range[0]) / range_y * screen_height)
    return screen_x, screen_y

def screen_to_km_coords(screen_x, screen_y, x_range, y_range, screen_width, screen_height):
    """Convert screen pixels to km coordinates"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    x_km = x_range[0] + (screen_x / screen_width) * range_x
    y_km = y_range[0] + (screen_y / screen_height) * range_y
    return x_km, y_km

# Cache for color calculations to avoid repeated HSV conversions
_COLOR_CACHE = {}

def _get_panel_color(power_pct):
    """Get cached panel color for given power percentage"""
    # Quantize to reduce cache size
    cache_key = int(power_pct * 100)
    
    if cache_key not in _COLOR_CACHE:
        if power_pct > 0.8:
            # High power - blue to green gradient
            h = 0.6 - (power_pct - 0.8) * 0.6 / 0.2
            s = 0.8
            v = 0.9
        else:
            # Lower power - green to red gradient
            h = 0.3 - (0.8 - power_pct) * 0.3 / 0.8
            s = 0.8
            v = 0.8
        
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        _COLOR_CACHE[cache_key] = (int(r * 255), int(g * 255), int(b * 255))
    
    return _COLOR_CACHE[cache_key]

def draw_solar_panels(screen, panel_df, panel_coverage, power_output, x_range, y_range, width, height):
    """
    Optimized solar panel drawing with vectorized coordinate conversion.
    
    Args:
        screen: Pygame surface to draw on
        panel_df: DataFrame with panel information
        panel_coverage: Dict mapping panel_id to coverage percentage
        power_output: Dict mapping panel_id to power output info
        x_range, y_range: Coordinate ranges in km
        width, height: Screen dimensions
    
    Returns:
        List of affected panel IDs sorted by coverage.
    """
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    panel_size_km = 0.4  # Default panel size
    panel_size_px = int(panel_size_km / range_x * width)
    
    # Early exit if panel size is too small
    if panel_size_px < 2:
        return []
    
    # Vectorized coordinate conversion for all panels
    x_km_array = panel_df['x_km'].values
    y_km_array = panel_df['y_km'].values
    panel_ids = panel_df['panel_id'].values
    
    x_px_array, y_px_array = km_to_screen_coords_vectorized(
        x_km_array, y_km_array, x_range, y_range, width, height
    )
    
    # Pre-compute power percentages for all panels
    power_percentages = np.zeros(len(panel_df))
    coverages = np.zeros(len(panel_df))
    
    for i, panel_id in enumerate(panel_ids):
        coverages[i] = panel_coverage.get(panel_id, 0.0)
        
        power_data = power_output.get(panel_id, {})
        power_value = power_data.get('final_power', 0)
        max_power = power_data.get('baseline', 1.0)
        
        power_percentages[i] = min(1.0, max(0.0, power_value / max_power if max_power > 0 else 0))
    
    # Collect affected panels efficiently
    affected_mask = coverages > 0
    affected_panels = [(panel_ids[i], coverages[i]) for i in np.where(affected_mask)[0]]
    
    # Sort affected panels by coverage
    affected_panels.sort(key=lambda x: x[1], reverse=True)
    
    # Batch drawing operations
    panel_rects = []
    panel_colors = []
    shadow_operations = []
    
    # Pre-calculate all drawing operations
    for i in range(len(panel_df)):
        x_px, y_px = x_px_array[i], y_px_array[i]
        panel_id = panel_ids[i]
        coverage = coverages[i]
        power_pct = power_percentages[i]
        
        # Skip panels outside screen bounds
        if (x_px < -panel_size_px or x_px > width + panel_size_px or 
            y_px < -panel_size_px or y_px > height + panel_size_px):
            continue
        
        # Get panel color (cached)
        panel_color = _get_panel_color(power_pct)
        
        # Create panel rect
        panel_rect = pygame.Rect(
            x_px - panel_size_px//2, 
            y_px - panel_size_px//2, 
            panel_size_px, 
            panel_size_px
        )
        
        panel_rects.append(panel_rect)
        panel_colors.append(panel_color)
        
        # Prepare shadow operations if needed
        if coverage > 0.05:
            shadow_size = int(panel_size_px * coverage)
            shadow_pos = (x_px - shadow_size//2, y_px - shadow_size//2)
            shadow_operations.append((shadow_size, shadow_pos))
        else:
            shadow_operations.append(None)
    
    # Batch draw all panels
    for i, (rect, color) in enumerate(zip(panel_rects, panel_colors)):
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, width=1)  # Border
        
        # Draw shadow if needed
        shadow_op = shadow_operations[i]
        if shadow_op is not None:
            shadow_size, shadow_pos = shadow_op
            shadow_surface = pygame.Surface((shadow_size, shadow_size), pygame.SRCALPHA)
            pygame.draw.rect(shadow_surface, (0, 0, 0, 120), (0, 0, shadow_size, shadow_size))
            screen.blit(shadow_surface, shadow_pos)
    
    # Draw panel IDs only if manageable number and large enough
    if len(panel_df) < 100 and panel_size_px > 15:
        font_size = 9 if len(panel_df) < 60 else 7
        font = pygame.font.SysFont('Arial', font_size)
        
        for i, rect in enumerate(panel_rects):
            if i < len(panel_ids):
                label = font.render(panel_ids[i], True, (255, 255, 255))
                label_rect = label.get_rect(center=rect.center)
                screen.blit(label, label_rect)
    
    return [panel_id for panel_id, _ in affected_panels]

def create_ui_element(text, position, size, font_size=16, bg_color=(255, 255, 255, 200), text_color=(0, 0, 0)):
    """Create a UI element with text"""
    surface = pygame.Surface(size, pygame.SRCALPHA)
    surface.fill(bg_color)
    
    font = pygame.font.SysFont('Arial', font_size)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(size[0]//2, size[1]//2))
    
    surface.blit(text_surface, text_rect)
    
    return surface


# ===== OpenGL Background Grid Renderer =====

class GLGridRenderer:
    """OpenGL-based grid renderer for background with optimized vertex generation"""
    
    # Vertex shader for grid lines
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec2 position;
    
    uniform mat4 projection;
    
    void main() {
        gl_Position = projection * vec4(position, 0.0, 1.0);
    }
    """
    
    # Fragment shader for grid lines
    FRAGMENT_SHADER = """
    #version 330 core
    out vec4 fragColor;
    
    uniform vec4 lineColor;
    
    void main() {
        fragColor = lineColor;
    }
    """
    
    def __init__(self, screen_size):
        """Initialize the OpenGL grid renderer
        
        Args:
            screen_size: Tuple of (width, height) in pixels
        """
        self.screen_size = screen_size
        self.initialized = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.projection_matrix = self._create_ortho_matrix(0, screen_size[0], screen_size[1], 0, -1, 1)
        
        # Cache for grid vertices to avoid regeneration
        self._grid_cache = {}
        
    def init_gl(self):
        """Initialize OpenGL resources for grid rendering"""
        if self.initialized:
            return True
            
        try:
            # Compile shaders
            self.shader_program = compileProgram(
                compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER),
                compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
            
            # Set up VAO
            self.vao = glGenVertexArrays(1)
            
            # Set up VBO (will be populated in render)
            self.vbo = glGenBuffers(1)
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing GL grid renderer: {e}")
            return False
        
    def _create_ortho_matrix(self, left, right, bottom, top, near, far):
        """Create an orthographic projection matrix"""
        width = right - left
        height = top - bottom
        depth = far - near
        
        return np.array([
            [2/width, 0, 0, -(right + left)/width],
            [0, 2/height, 0, -(top + bottom)/height],
            [0, 0, -2/depth, -(far + near)/depth],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    def _generate_grid_vertices(self, x_range, y_range, width, height):
        """Generate grid vertices with caching"""
        cache_key = (x_range[0], x_range[1], y_range[0], y_range[1], width, height)
        
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]
        
        range_x = x_range[1] - x_range[0]
        range_y = y_range[1] - y_range[0]
        
        # Determine grid spacing based on view range
        grid_interval_km = _get_grid_spacing(range_x)
        
        # Vectorized vertex generation
        x_km_lines = np.arange(x_range[0], x_range[1] + grid_interval_km, grid_interval_km)
        y_km_lines = np.arange(y_range[0], y_range[1] + grid_interval_km, grid_interval_km)
        
        # Convert to pixel coordinates
        x_px_lines = ((x_km_lines - x_range[0]) / range_x * width).astype(np.float32)
        y_px_lines = ((y_km_lines - y_range[0]) / range_y * height).astype(np.float32)
        
        # Generate vertices for vertical lines
        v_vertices = np.column_stack([
            np.repeat(x_px_lines, 2),
            np.tile([0, height], len(x_px_lines))
        ]).astype(np.float32)
        
        # Generate vertices for horizontal lines
        h_vertices = np.column_stack([
            np.tile([0, width], len(y_px_lines)),
            np.repeat(y_px_lines, 2)
        ]).astype(np.float32)
        
        # Combine all vertices
        vertices = np.vstack([v_vertices, h_vertices])
        
        # Cache the result
        self._grid_cache[cache_key] = vertices
        
        # Limit cache size
        if len(self._grid_cache) > 10:
            oldest_key = next(iter(self._grid_cache))
            del self._grid_cache[oldest_key]
        
        return vertices
    
    def render_grid(self, x_range, y_range, width, height, grid_color=(0.8, 0.8, 0.8, 1.0)):
        """Render background grid with OpenGL using cached vertices
        
        Args:
            x_range, y_range: Coordinate ranges in km
            width, height: Screen dimensions
            grid_color: Color for grid lines (RGBA, 0-1 range)
        """
        if not self.initialized and not self.init_gl():
            print("Failed to initialize OpenGL for grid. Skipping rendering.")
            return
        
        # Get cached vertices
        vertices = self._generate_grid_vertices(x_range, y_range, width, height)
        
        # Prepare for rendering
        glUseProgram(self.shader_program)
        
        # Set projection matrix uniform
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection_matrix)
        
        # Set line color uniform
        color_loc = glGetUniformLocation(self.shader_program, "lineColor")
        glUniform4fv(color_loc, 1, grid_color)
        
        # Bind VAO
        glBindVertexArray(self.vao)
        
        # Update VBO with vertices
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Set vertex attributes
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Draw grid lines
        glDrawArrays(GL_LINES, 0, len(vertices))
        
        # Clean up
        glBindVertexArray(0)
        glUseProgram(0)


# ===== OpenGL Panel Renderer =====

class GLPanelRenderer:
    """OpenGL-based renderer for solar panels with optimized instanced rendering"""
    
    # Vertex shader for panels with instancing
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texCoords;
    layout(location = 2) in vec2 instancePos;
    layout(location = 3) in float instancePower;
    layout(location = 4) in float instanceCoverage;
    
    out vec2 fragTexCoords;
    out vec2 panelCoord;
    flat out float powerPercentage;
    flat out float coverage;
    
    uniform mat4 projection;
    uniform float panelSize;
    uniform float time;
    
    void main() {
        vec2 worldPos = instancePos + position * panelSize;
        gl_Position = projection * vec4(worldPos, 0.0, 1.0);
        fragTexCoords = texCoords;
        panelCoord = position;
        powerPercentage = instancePower;
        coverage = instanceCoverage;
    }
    """
    
    # Fragment shader for panels with power-based coloring and shadow effects
    FRAGMENT_SHADER = """
    #version 330 core
    in vec2 fragTexCoords;
    in vec2 panelCoord;
    flat in float powerPercentage;
    flat in float coverage;
    
    out vec4 fragColor;
    
    uniform float time;
    
    // Gradient colors for power levels
    vec3 highPowerColor = vec3(0.0, 0.0, 1.0);    // Blue
    vec3 medPowerColor = vec3(0.0, 0.8, 0.0);     // Green
    vec3 lowPowerColor = vec3(1.0, 0.0, 0.0);     // Red
    
    void main() {
        // Panel border effect
        float border = 0.03;
        bool isBorder = abs(fragTexCoords.x - 0.5) > (0.5 - border) || 
                       abs(fragTexCoords.y - 0.5) > (0.5 - border);
        
        // Calculate base color based on power percentage
        vec3 baseColor;
        if (powerPercentage > 0.8) {
            // High power - blue to green gradient
            float t = (powerPercentage - 0.8) / 0.2;
            baseColor = mix(medPowerColor, highPowerColor, t);
        } else {
            // Lower power - green to red gradient
            float t = powerPercentage / 0.8;
            baseColor = mix(lowPowerColor, medPowerColor, t);
        }
        
        // Apply solar panel cell pattern
        float cellSize = 0.2;
        float cellX = mod(fragTexCoords.x / cellSize, 1.0);
        float cellY = mod(fragTexCoords.y / cellSize, 1.0);
        float cellBorder = 0.05;
        bool isGridline = cellX < cellBorder || cellY < cellBorder;
        
        // Darkened gridlines
        if (isGridline) {
            baseColor *= 0.7;
        }
        
        // Apply border
        vec3 borderColor = vec3(0.1, 0.1, 0.1);
        if (isBorder) {
            baseColor = borderColor;
        }
        
        // Apply cloud shadow effect
        float shadowIntensity = coverage * 0.8;
        if (coverage > 0.05) {
            baseColor *= (1.0 - shadowIntensity);
        }
        
        fragColor = vec4(baseColor, 1.0);
    }
    """
    
    def __init__(self, screen_size):
        """Initialize the OpenGL panel renderer
        
        Args:
            screen_size: Tuple of (width, height) in pixels
        """
        self.screen_size = screen_size
        self.initialized = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.instance_vbo = None
        self.projection_matrix = self._create_ortho_matrix(0, screen_size[0], screen_size[1], 0, -1, 1)
        self.start_time = pygame.time.get_ticks() / 1000.0
    
    def init_gl(self):
        """Initialize OpenGL context and resources"""
        if self.initialized:
            return True
            
        # Compile shaders
        try:
            self.shader_program = compileProgram(
                compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER),
                compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"Error compiling panel shaders: {e}")
            return False
        
        # Create a quad for panels
        vertices = np.array([
            # x, y, u, v
            -0.5, -0.5, 0.0, 0.0,  # Bottom left
             0.5, -0.5, 1.0, 0.0,  # Bottom right
             0.5,  0.5, 1.0, 1.0,  # Top right
            -0.5,  0.5, 0.0, 1.0   # Top left
        ], dtype=np.float32)
        
        # Indices for quad
        indices = np.array([
            0, 1, 2,  # First triangle
            2, 3, 0   # Second triangle
        ], dtype=np.uint32)
        
        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Create and bind VBO for quad vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)