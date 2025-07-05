"""
Optimized Cloud Renderer for Solar Farm Simulation
Optimized for batch operations, pre-computed textures, and GPU instancing
"""
import pygame
import numpy as np
import math
import colorsys
from pygame.locals import *

# OpenGL imports - only imported if OpenGL rendering is used
try:
    from OpenGL.GL import *
    from OpenGL.GL import (
        GL_ELEMENT_ARRAY_BUFFER, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
        glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
        glBufferData, glVertexAttribPointer, glEnableVertexAttribArray,
        glVertexAttribDivisor, glVertexAttribIPointer, glUseProgram,
        glGetUniformLocation, glUniformMatrix4fv, glUniform1f, glActiveTexture,
        glBindTexture, glUniform1i, glDrawElementsInstanced, glViewport,
        glClearColor, glEnable, glBlendFunc, glClear, GL_ARRAY_BUFFER,
        GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_FLOAT, GL_FALSE, GL_INT,
        GL_TRIANGLES, GL_UNSIGNED_INT, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE,
        GL_TEXTURE_WRAP_T, GL_TEXTURE_MIN_FILTER, GL_LINEAR, GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE0, GL_TEXTURE1, GL_R8, GL_RED, GL_REPEAT, GL_COLOR_BUFFER_BIT,
        glTexImage2D, glGenTextures, glTexParameteri  # Added glTexParameteri here
    )
    from OpenGL.GL.shaders import compileProgram, compileShader
    import ctypes
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Using Pygame renderer only.")

# Global caches for optimization
_NOISE_TEXTURES = {}
_CLOUD_SURFACE_CACHE = {}
_COORDINATE_CACHE = {}

def km_to_screen_coords_vectorized(x_km_array, y_km_array, x_range, y_range, screen_width, screen_height):
    """Vectorized version of km_to_screen_coords for multiple coordinates"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    
    screen_x = ((x_km_array - x_range[0]) / range_x * screen_width).astype(int)
    screen_y = ((y_km_array - y_range[0]) / range_y * screen_height).astype(int)
    
    return screen_x, screen_y

def km_to_screen_coords(x_km, y_km, x_range, y_range, screen_width, screen_height):
    """Convert kilometer coordinates to screen pixels (single point)"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    screen_x = int((x_km - x_range[0]) / range_x * screen_width)
    screen_y = int((y_km - y_range[0]) / range_y * screen_height)
    return screen_x, screen_y

def _generate_noise_texture(size, noise_type='perlin'):
    """Generate and cache noise textures for cloud rendering"""
    cache_key = (size, noise_type)
    
    if cache_key in _NOISE_TEXTURES:
        return _NOISE_TEXTURES[cache_key]
    
    # Generate noise using vectorized operations
    if noise_type == 'perlin':
        # Simple pseudo-Perlin noise using vectorized operations
        x = np.arange(size)
        y = np.arange(size)
        X, Y = np.meshgrid(x, y)
        
        # Multi-octave noise
        noise = np.zeros((size, size))
        
        # Octave 1 (coarse)
        scale1 = 0.1
        noise += 0.5 * np.sin(X * scale1) * np.cos(Y * scale1)
        
        # Octave 2 (medium)
        scale2 = 0.3
        noise += 0.3 * np.sin(X * scale2 + 1.5) * np.cos(Y * scale2 + 2.1)
        
        # Octave 3 (fine)
        scale3 = 0.7
        noise += 0.2 * np.sin(X * scale3 + 3.7) * np.cos(Y * scale3 + 4.2)
        
        # Normalize to 0-1 range
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
    elif noise_type == 'turbulence':
        # Turbulence noise pattern
        x = np.linspace(0, 4*np.pi, size)
        y = np.linspace(0, 4*np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        noise = 0.5 * (np.sin(X) + np.cos(Y) + 
                      0.5 * np.sin(2*X) + 0.5 * np.cos(2*Y) +
                      0.25 * np.sin(4*X) + 0.25 * np.cos(4*Y))
        
        # Normalize to 0-1 range
        noise = (noise + 2.25) / 4.5
        
    else:  # 'random'
        noise = np.random.random((size, size))
    
    # Cache the result
    _NOISE_TEXTURES[cache_key] = noise
    
    # Limit cache size
    if len(_NOISE_TEXTURES) > 20:
        oldest_key = next(iter(_NOISE_TEXTURES))
        del _NOISE_TEXTURES[oldest_key]
    
    return noise

def _create_cloud_base_surface(width, height, cloud_type="cumulus", altitude=1.0):
    """Create a base cloud surface using vectorized operations"""
    cache_key = (width, height, cloud_type, int(altitude * 10))
    
    if cache_key in _CLOUD_SURFACE_CACHE:
        return _CLOUD_SURFACE_CACHE[cache_key].copy()
    
    # Create coordinate grids
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    # Calculate distance from center (vectorized)
    center_x = width // 2
    center_y = height // 2
    dx = (X - center_x) / (width / 2)
    dy = (Y - center_y) / (height / 2)
    distance = np.sqrt(dx**2 + dy**2)
    
    # Create elliptical mask
    ellipse_mask = distance <= 1.0
    
    # Apply cloud type specific parameters
    if cloud_type == "cirrus":
        base_color = (250, 250, 255)
        noise_scale = 0.3
        edge_softness = 0.8
    elif cloud_type == "cumulonimbus":
        base_color = (240, 240, 245)
        noise_scale = 0.1
        edge_softness = 0.3
    else:  # cumulus
        base_color = (245, 245, 250)
        noise_scale = 0.2
        edge_softness = 0.5
    
    # Adjust for altitude
    if altitude > 5.0:
        brightness = min(1.0, 0.98 + altitude * 0.004)
        base_color = tuple(min(255, int(c * brightness)) for c in base_color)
    elif altitude < 1.0:
        darkness = max(0.92, 1.0 - (1.0 - altitude) * 0.1)
        base_color = tuple(int(c * darkness) for c in base_color)
    
    # Generate noise texture for this size
    noise_size = min(256, max(width, height))
    noise = _generate_noise_texture(noise_size, 'perlin')
    
    # Resize noise to match cloud size if needed
    if noise_size != width or noise_size != height:
        # Simple bilinear interpolation for noise
        noise_x = np.linspace(0, noise_size-1, width)
        noise_y = np.linspace(0, noise_size-1, height)
        noise_X, noise_Y = np.meshgrid(noise_x, noise_y)
        
        # Interpolate noise values
        noise_X_int = noise_X.astype(int)
        noise_Y_int = noise_Y.astype(int)
        noise_X_int = np.clip(noise_X_int, 0, noise_size-1)
        noise_Y_int = np.clip(noise_Y_int, 0, noise_size-1)
        
        interpolated_noise = noise[noise_Y_int, noise_X_int]
    else:
        interpolated_noise = noise
    
    # Apply noise to distance calculation
    noisy_distance = distance + interpolated_noise * noise_scale
    
    # Create gradient alpha with soft edges
    alpha_values = np.where(ellipse_mask, 
                           np.maximum(0, 1.0 - noisy_distance**2) * edge_softness, 
                           0)
    
    # Add depth gradient (darker at bottom)
    depth_gradient = 1.0 - dy * 0.1
    depth_gradient = np.clip(depth_gradient, 0.8, 1.0)
    
    # Create the surface
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    
    # Vectorized color application
    r_values = (base_color[0] * depth_gradient).astype(np.uint8)
    g_values = (base_color[1] * depth_gradient).astype(np.uint8)
    b_values = (base_color[2] * depth_gradient).astype(np.uint8)
    a_values = (alpha_values * 255).astype(np.uint8)
    
    # Create RGBA array
    rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_array[:, :, 0] = r_values
    rgba_array[:, :, 1] = g_values
    rgba_array[:, :, 2] = b_values
    rgba_array[:, :, 3] = a_values

    # Fix: Use frombuffer to create surface from RGBA array
    surface = pygame.image.frombuffer(rgba_array.tobytes(), (width, height), 'RGBA').convert_alpha()

    # Cache the result
    _CLOUD_SURFACE_CACHE[cache_key] = surface.copy()

    # Limit cache size
    if len(_CLOUD_SURFACE_CACHE) > 50:
        oldest_key = next(iter(_CLOUD_SURFACE_CACHE))
        del _CLOUD_SURFACE_CACHE[oldest_key]

    return surface

def create_cloud_surface(ellipse_params, domain_size, area_size_km, width, height, x_range, y_range):
    """
    Optimized cloud surface creation using pre-computed textures and vectorized operations.
    
    Args:
        ellipse_params: Tuple with cloud parameters (cx, cy, cw, ch, crot, cop, [alt], [type])
        domain_size: Size of simulation domain in meters
        area_size_km: Size of simulation area in kilometers
        width, height: Screen dimensions
        x_range, y_range: Coordinate ranges in km
        
    Returns:
        Tuple of (cloud_surface, position)
    """
    # Extract parameters
    if len(ellipse_params) >= 7:
        cx, cy, cw, ch, crot, cop, altitude = ellipse_params[:7]
        cloud_type = ellipse_params[7] if len(ellipse_params) > 7 else "cumulus"
    else:
        cx, cy, cw, ch, crot, cop = ellipse_params
        altitude = 1.0
        cloud_type = "cumulus"
    
    # Convert to km (vectorized where possible)
    scale_factor = area_size_km / domain_size
    cx_km = cx * scale_factor
    cy_km = cy * scale_factor
    cw_km = cw * scale_factor
    ch_km = ch * scale_factor
    
    # Convert to screen coordinates
    cx_px, cy_px = km_to_screen_coords(cx_km, cy_km, x_range, y_range, width, height)
    
    # Calculate size in pixels
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    cw_px = max(10, int(cw_km / range_x * width))
    ch_px = max(10, int(ch_km / range_y * height))
    
    # Create base cloud surface using optimized method
    base_surface = _create_cloud_base_surface(cw_px, ch_px, cloud_type, altitude)
    
    # Apply opacity scaling
    opacity_factor = min(0.95, cop)
    if opacity_factor < 1.0:
        # Apply opacity to entire surface
        opacity_surface = pygame.Surface((cw_px, ch_px), pygame.SRCALPHA)
        opacity_surface.set_alpha(int(opacity_factor * 255))
        opacity_surface.blit(base_surface, (0, 0))
        cloud_surface = opacity_surface
    else:
        cloud_surface = base_surface
    
    # Calculate position (top-left corner for blitting)
    pos = (cx_px - cw_px // 2, cy_px - ch_px // 2)
    
    # Apply rotation if needed (optimized)
    if abs(crot) > 0.01:  # Only rotate if significant rotation
        rot_degrees = math.degrees(crot)
        cloud_surface = pygame.transform.rotate(cloud_surface, rot_degrees)
        
        # Adjust position for rotation
        rot_rect = cloud_surface.get_rect(center=(cx_px, cy_px))
        pos = rot_rect.topleft
    
    return cloud_surface, pos

def draw_cloud_trail_batch(screen, all_cloud_positions, x_range, y_range, width, height):
    """
    Optimized cloud trail drawing using batch operations for multiple clouds.
    
    Args:
        screen: Pygame surface to draw on
        all_cloud_positions: List of cloud position lists [(x_km, y_km), ...]
        x_range, y_range: Coordinate ranges in km
        width, height: Screen dimensions
    """
    if not all_cloud_positions:
        return
    
    # Collect all trail segments for batch processing
    all_segments = []
    
    for cloud_positions in all_cloud_positions:
        if not cloud_positions or len(cloud_positions) < 2:
            continue
        
        # Convert all positions to screen coordinates (vectorized)
        positions_array = np.array(cloud_positions)
        x_km_array = positions_array[:, 0]
        y_km_array = positions_array[:, 1]
        
        x_px_array, y_px_array = km_to_screen_coords_vectorized(
            x_km_array, y_km_array, x_range, y_range, width, height
        )
        
        # Create segments with alpha values
        for i in range(len(x_px_array) - 1):
            start_pos = (x_px_array[i], y_px_array[i])
            end_pos = (x_px_array[i + 1], y_px_array[i + 1])
            
            # Calculate alpha based on position in history
            alpha = int(255 * (i / len(x_px_array)))
            
            all_segments.append((start_pos, end_pos, alpha))
    
    # Batch draw all segments
    if all_segments:
        # Group segments by alpha for more efficient drawing
        alpha_groups = {}
        for start_pos, end_pos, alpha in all_segments:
            if alpha not in alpha_groups:
                alpha_groups[alpha] = []
            alpha_groups[alpha].append((start_pos, end_pos))
        
        # Draw each alpha group
        for alpha, segments in alpha_groups.items():
            color = (200, 200, 200, alpha)
            for start_pos, end_pos in segments:
                pygame.draw.line(screen, color, start_pos, end_pos, width=2)

def draw_cloud_trail(screen, cloud_positions, x_range, y_range, width, height):
    """
    Optimized single cloud trail drawing.
    
    Args:
        screen: Pygame surface to draw on
        cloud_positions: List of (x_km, y_km) tuples
        x_range, y_range: Coordinate ranges in km
        width, height: Screen dimensions
    """
    if not cloud_positions or len(cloud_positions) < 2:
        return
    
    # Use batch method for single cloud
    draw_cloud_trail_batch(screen, [cloud_positions], x_range, y_range, width, height)

def create_cloud_surfaces_batch(cloud_ellipses_list, domain_size, area_size_km, width, height, x_range, y_range):
    """
    Batch create cloud surfaces for multiple clouds to amortize overhead.
    
    Args:
        cloud_ellipses_list: List of cloud ellipse parameters
        domain_size: Size of simulation domain in meters
        area_size_km: Size of simulation area in kilometers
        width, height: Screen dimensions
        x_range, y_range: Coordinate ranges in km
        
    Returns:
        List of (cloud_surface, position) tuples
    """
    if not cloud_ellipses_list:
        return []
    
    results = []
    
    # Group clouds by similar characteristics for cache efficiency
    cloud_groups = {}
    
    for i, ellipse_params in enumerate(cloud_ellipses_list):
        # Extract type and altitude for grouping
        if len(ellipse_params) >= 7:
            altitude = ellipse_params[6]
            cloud_type = ellipse_params[7] if len(ellipse_params) > 7 else "cumulus"
        else:
            altitude = 1.0
            cloud_type = "cumulus"
        
        group_key = (cloud_type, int(altitude * 10))
        if group_key not in cloud_groups:
            cloud_groups[group_key] = []
        cloud_groups[group_key].append((i, ellipse_params))
    
    # Process each group
    result_dict = {}
    
    for group_key, group_clouds in cloud_groups.items():
        for original_index, ellipse_params in group_clouds:
            surface, pos = create_cloud_surface(
                ellipse_params, domain_size, area_size_km, 
                width, height, x_range, y_range
            )
            result_dict[original_index] = (surface, pos)
    
    # Restore original order
    for i in range(len(cloud_ellipses_list)):
        results.append(result_dict[i])
    
    return results


# ===== OpenGL Rendering Support with Instanced Rendering =====

class GLCloudRenderer:
    """Optimized OpenGL cloud renderer with instanced rendering and texture atlases"""
    
    # Vertex shader with instancing support
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texCoords;
    layout(location = 2) in vec2 instancePos;
    layout(location = 3) in vec2 instanceSize;
    layout(location = 4) in float instanceOpacity;
    layout(location = 5) in float instanceAltitude;
    layout(location = 6) in int instanceType;
    layout(location = 7) in float instanceRotation;
    
    out vec2 fragTexCoords;
    flat out float opacity;
    flat out float altitude;
    flat out int cloudType;
    
    uniform mat4 projection;
    uniform vec2 prevPosition;
    uniform vec2 velocity;
    uniform float u_alpha;
    
    void main() {
        // Apply rotation
        float s = sin(instanceRotation);
        float c = cos(instanceRotation);
        vec2 rotatedPos = vec2(
            position.x * c - position.y * s,
            position.x * s + position.y * c
        );
        
        // Scale and translate
        vec2 worldPos = instancePos + rotatedPos * instanceSize;
        
        gl_Position = projection * vec4(worldPos, 0.0, 1.0);
        fragTexCoords = texCoords;
        opacity = instanceOpacity;
        altitude = instanceAltitude;
        cloudType = instanceType;
    }
    """
    
    # Optimized fragment shader
    FRAGMENT_SHADER = """
    #version 330 core
    in vec2 fragTexCoords;
    flat in float opacity;
    flat in float altitude;
    flat in int cloudType;
    
    out vec4 fragColor;
    
    uniform sampler2D cloudTexture;
    uniform sampler2D noiseTexture;
    
    void main() {
        // Sample base cloud texture
        vec4 cloudSample = texture(cloudTexture, fragTexCoords);
        
        // Sample noise for variation
        vec4 noiseSample = texture(noiseTexture, fragTexCoords * 3.0);
        
        // Calculate distance from center for falloff
        vec2 center = vec2(0.5, 0.5);
        float dist = distance(fragTexCoords, center);
        
        // Base cloud shape with soft edges
        float alpha = smoothstep(0.6, 0.2, dist) * opacity;
        
        // Apply noise variation
        alpha *= (0.8 + 0.2 * noiseSample.r);
        
        // Cloud type effects
        vec3 finalColor;
        
        if (cloudType == 0) {  // Cirrus
            finalColor = vec3(1.0, 1.0, 1.0);
            // Add wispy effect
            float wispy = sin(fragTexCoords.x * 15.0 + noiseSample.g * 3.0) * 0.5 + 0.5;
            alpha *= wispy * 0.6 + 0.4;
        }
        else if (cloudType == 1) {  // Cumulus
            finalColor = vec3(0.95, 0.95, 0.95);
        }
        else {  // Cumulonimbus
            finalColor = vec3(0.85, 0.85, 0.9);
            alpha *= 1.3;  // Denser
        }
        
        // Altitude-based brightness
        float altitudeFactor = min(1.0, altitude / 5.0);
        finalColor += vec3(0.1 * altitudeFactor * (1.0 - dist));
        
        // Output final color
        fragColor = vec4(finalColor, alpha);
    }
    """
    
    def __init__(self, screen_size):
        """Initialize the optimized OpenGL cloud renderer"""
        self.screen_size = screen_size
        self.initialized = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.instance_vbo = None
        self.projection_matrix = self._create_ortho_matrix(0, screen_size[0], screen_size[1], 0, -1, 1)
        
        # Texture resources
        self.cloud_texture = None
        self.noise_texture = None
        
        # Instance data buffer
        self.max_instances = 1000
    
    def init_gl(self):
        """Initialize OpenGL context and resources"""
        if self.initialized:
            return True
            
        try:
            # Compile shaders
            self.shader_program = compileProgram(
                compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER),
                compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(f"Error compiling shaders: {e}")
            return False
        
        # Create quad vertices
        vertices = np.array([
            # x, y, u, v
            -0.5, -0.5, 0.0, 0.0,
             0.5, -0.5, 1.0, 0.0,
             0.5,  0.5, 1.0, 1.0,
            -0.5,  0.5, 0.0, 1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Set up VAO and VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Quad vertices
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Position and texture coordinates
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))
        glEnableVertexAttribArray(1)
        
        # Instance buffer
        self.instance_vbo = glGenBuffers(1)
        
        # Element buffer
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Create textures
        self.cloud_texture = self._create_optimized_cloud_texture()
        self.noise_texture = self._create_noise_texture()
        
        # OpenGL state
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glBindVertexArray(0)
        
        self.initialized = True
        return True
    
    def _create_optimized_cloud_texture(self, size=512):
        """Create an optimized cloud texture using pre-computed noise"""
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        # Use cached noise for texture generation
        noise = _generate_noise_texture(size, 'perlin')
        
        # Create cloud texture data
        data = np.zeros((size, size, 4), dtype=np.uint8)
        
        # Create radial gradient
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center)**2 + (y - center)**2) / center
        
        # Apply noise to create cloud-like structure
        cloud_alpha = np.maximum(0, 1 - dist + noise * 0.3)
        cloud_alpha = np.clip(cloud_alpha, 0, 1)
        
        # Set texture data
        data[:, :, 0] = 255  # R
        data[:, :, 1] = 255  # G
        data[:, :, 2] = 255  # B
        data[:, :, 3] = (cloud_alpha * 255).astype(np.uint8)  # A
        
        # Upload texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        return texture
    
    def _create_noise_texture(self, size=256):
        """Create a noise texture for variation"""
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        # Generate noise
        noise = _generate_noise_texture(size, 'turbulence')
        
        # Convert to texture data
        data = (noise * 255).astype(np.uint8)
        
        # Upload as single-channel texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, size, size, 0, GL_RED, GL_UNSIGNED_BYTE, data)
        
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        return texture
    
    def _create_ortho_matrix(self, left, right, bottom, top, near, far):
        """Create orthographic projection matrix"""
        width = right - left
        height = top - bottom
        depth = far - near
        
        return np.array([
            [2/width, 0, 0, -(right + left)/width],
            [0, 2/height, 0, -(top + bottom)/height],
            [0, 0, -2/depth, -(far + near)/depth],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    def render_clouds_instanced(self, cloud_ellipses, x_range, y_range, width, height, alpha=0.0):
        """Render multiple clouds using instanced rendering"""
        if not self.initialized and not self.init_gl():
            print("Failed to initialize OpenGL. Skipping cloud rendering.")
            return
        
        if not cloud_ellipses:
            return
        
        # Prepare instance data
        num_clouds = min(len(cloud_ellipses), self.max_instances)
        instance_data = np.zeros((num_clouds, 8), dtype=np.float32)
        
        range_x = x_range[1] - x_range[0]
        range_y = y_range[1] - y_range[0]
        
        for i, ellipse_params in enumerate(cloud_ellipses[:num_clouds]):
            if len(ellipse_params) >= 7:
                cx, cy, cw, ch, crot, cop, altitude = ellipse_params[:7]
                cloud_type = ellipse_params[7] if len(ellipse_params) > 7 else "cumulus"
            else:
                cx, cy, cw, ch, crot, cop = ellipse_params
                altitude = 1.0
                cloud_type = "cumulus"
            
            # Convert to screen coordinates
            cx_km = cx / 1000
            cy_km = cy / 1000
            cw_km = cw / 1000
            ch_km = ch / 1000
            
            screen_x = (cx_km - x_range[0]) / range_x * width
            screen_y = (cy_km - y_range[0]) / range_y * height
            screen_w = cw_km / range_x * width
            screen_h = ch_km / range_y * height
            
            # Map cloud type to integer
            if cloud_type == "cirrus":
                type_int = 0
            elif cloud_type == "cumulonimbus":
                type_int = 2
            else:  # cumulus
                type_int = 1
            
            # Fill instance data
            instance_data[i] = [
                screen_x, screen_y,        # Position
                screen_w, screen_h,        # Size
                cop,                       # Opacity
                altitude,                  # Altitude
                type_int,                  # Cloud type
                crot                       # Rotation
            ]
        
        # Upload instance data
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_vbo)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        # Set instance attributes
        stride = 8 * instance_data.itemsize
        
        # Instance position
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)
        glVertexAttribDivisor(2, 1)
        
        # Instance size
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * instance_data.itemsize))
        glEnableVertexAttribArray(3)
        glVertexAttribDivisor(3, 1)
        
        # Instance opacity
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(4 * instance_data.itemsize))
        glEnableVertexAttribArray(4)
        glVertexAttribDivisor(4, 1)
        
        # Instance altitude
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(5 * instance_data.itemsize))
        glEnableVertexAttribArray(5)
        glVertexAttribDivisor(5, 1)
        
        # Instance type
        glVertexAttribIPointer(6, 1, GL_INT, stride, ctypes.c_void_p(6 * instance_data.itemsize))
        glEnableVertexAttribArray(6)
        glVertexAttribDivisor(6, 1)
        
        # Instance rotation
        glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(7 * instance_data.itemsize))
        glEnableVertexAttribArray(7)
        glVertexAttribDivisor(7, 1)
        
        # Render
        glUseProgram(self.shader_program)
        
        # Set uniforms
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection_matrix)
        
        alpha_loc = glGetUniformLocation(self.shader_program, "u_alpha")
        glUniform1f(alpha_loc, alpha)
        
        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.cloud_texture)
        glUniform1i(glGetUniformLocation(self.shader_program, "cloudTexture"), 0)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.noise_texture)
        glUniform1i(glGetUniformLocation(self.shader_program, "noiseTexture"), 1)
        
        # Draw all instances
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None, num_clouds)
        
        # Cleanup
        glBindVertexArray(0)
        glUseProgram(0)
    
    def render_clouds(self, cloud_ellipses, x_range, y_range, width, height, alpha=0.0):
        """Wrapper for backward compatibility"""
        self.render_clouds_instanced(cloud_ellipses, x_range, y_range, width, height, alpha)


def initialize_gl_for_pygame(screen):
    """Initialize OpenGL for use with Pygame
    
    Args:
        screen: Pygame screen surface
    
    Returns:
        GLCloudRenderer instance or None if OpenGL initialization failed
    """
    if not OPENGL_AVAILABLE:
        print("OpenGL not available, cannot initialize GL renderer")
        return None
        
    width, height = screen.get_size()
    
    try:
        # Set Pygame to use OpenGL
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
        
        # Set up viewport
        glViewport(0, 0, width, height)
        
        # Set clear color
        glClearColor(0.9, 0.95, 1.0, 1.0)
        
        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create renderer
        renderer = GLCloudRenderer((width, height))
        if not renderer.init_gl():
            print("Failed to initialize GL renderer")
            return None
            
        return renderer
    except Exception as e:
        print(f"Error initializing OpenGL: {e}")
        return None


def render_gl_clouds(screen, cloud_renderer, cloud_ellipses, x_range, y_range, alpha=0.0):
    """Render clouds using OpenGL with instanced rendering
    
    Args:
        screen: Pygame screen surface
        cloud_renderer: GLCloudRenderer instance
        cloud_ellipses: List of cloud ellipse parameters
        x_range, y_range: Coordinate ranges in km
        alpha: Interpolation factor for smooth movement
    """
    if cloud_renderer is None:
        print("No GL renderer available, skipping GL cloud rendering")
        return
        
    width, height = screen.get_size()
    
    # Clear screen
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Render clouds using instanced rendering
    cloud_renderer.render_clouds_instanced(cloud_ellipses, x_range, y_range, width, height, alpha)
    
    # Swap buffers
    pygame.display.flip()


# ===== Cache Management Functions =====

def clear_cloud_caches():
    """Clear all cloud rendering caches to free memory"""
    global _NOISE_TEXTURES, _CLOUD_SURFACE_CACHE, _COORDINATE_CACHE
    
    _NOISE_TEXTURES.clear()
    _CLOUD_SURFACE_CACHE.clear()
    _COORDINATE_CACHE.clear()
    
    print("Cleared cloud rendering caches")

def get_cache_stats():
    """Get statistics about cache usage"""
    return {
        'noise_textures': len(_NOISE_TEXTURES),
        'cloud_surfaces': len(_CLOUD_SURFACE_CACHE),
        'coordinates': len(_COORDINATE_CACHE)
    }

# ===== Headless Mode Support =====

def create_headless_cloud_data(ellipse_params, domain_size, area_size_km):
    """
    Create minimal cloud data for headless mode (no visual rendering).
    
    Args:
        ellipse_params: Cloud parameters
        domain_size: Simulation domain size
        area_size_km: Area size in km
    
    Returns:
        Dictionary with essential cloud data
    """
    if len(ellipse_params) >= 7:
        cx, cy, cw, ch, crot, cop, altitude = ellipse_params[:7]
        cloud_type = ellipse_params[7] if len(ellipse_params) > 7 else "cumulus"
    else:
        cx, cy, cw, ch, crot, cop = ellipse_params
        altitude = 1.0
        cloud_type = "cumulus"
    
    # Convert to km
    scale_factor = area_size_km / domain_size
    
    return {
        'center_km': (cx * scale_factor, cy * scale_factor),
        'size_km': (cw * scale_factor, ch * scale_factor),
        'rotation': crot,
        'opacity': cop,
        'altitude': altitude,
        'type': cloud_type,
        'area': math.pi * (cw * scale_factor / 2) * (ch * scale_factor / 2)
    }

def batch_create_headless_cloud_data(cloud_ellipses_list, domain_size, area_size_km):
    """Batch create headless cloud data for multiple clouds"""
    return [create_headless_cloud_data(params, domain_size, area_size_km) 
            for params in cloud_ellipses_list]

def draw_cloud_debug_overlay(screen, cloud_ellipses, x_range, y_range, width, height):
    """Draws a debug marker (red circle) at each cloud's center position."""
    for ellipse in cloud_ellipses:
        if len(ellipse) >= 2:
            cx, cy = ellipse[0], ellipse[1]
            # Convert from simulation meters to km
            cx_km = cx / 1000.0
            cy_km = cy / 1000.0
            # Convert to screen coordinates
            screen_x = int((cx_km - x_range[0]) / (x_range[1] - x_range[0]) * width)
            screen_y = int((cy_km - y_range[0]) / (y_range[1] - y_range[0]) * height)
            pygame.draw.circle(screen, (255, 0, 0), (screen_x, screen_y), 8, 2)

def draw_clouds_pygame_fallback(screen, cloud_ellipses, x_range, y_range, width, height):
    """
    Draw clouds as simple white ellipses and overlay debug circles for each parcel.
    This function does NOT use OpenGL and will always work with Pygame.
    """
    for ellipse in cloud_ellipses:
        if len(ellipse) >= 7:
            cx, cy, cw, ch, crot, cop, alt = ellipse[:7]
        else:
            cx, cy, cw, ch, crot, cop = ellipse
            alt = 1.0
        # Convert from meters to km
        cx_km = cx / 1000.0
        cy_km = cy / 1000.0
        cw_km = cw / 1000.0
        ch_km = ch / 1000.0
        # Convert to screen coordinates
        screen_x = int((cx_km - x_range[0]) / (x_range[1] - x_range[0]) * width)
        screen_y = int((cy_km - y_range[0]) / (y_range[1] - y_range[0]) * height)
        screen_w = max(10, int(cw_km / (x_range[1] - x_range[0]) * width))
        screen_h = max(10, int(ch_km / (y_range[1] - y_range[0]) * height))
        # Draw ellipse (cloud)
        color = (255, 255, 255, int(min(255, cop * 255)))
        s = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        pygame.draw.ellipse(s, color, (0, 0, screen_w, screen_h))
        s = pygame.transform.rotate(s, math.degrees(crot))
        rect = s.get_rect(center=(screen_x, screen_y))
        screen.blit(s, rect)
        # Draw debug circle
        pygame.draw.circle(screen, (255, 0, 0), (screen_x, screen_y), 8, 2)
    # Optionally: print debug info if no clouds are visible
    if not cloud_ellipses:
        font = pygame.font.SysFont('Arial', 24)
        text = font.render('No clouds to display (fallback renderer)', True, (255, 0, 0))
        screen.blit(text, (20, 20))