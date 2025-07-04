import pygame
import numpy as np
import math
import colorsys
from pygame.locals import *
import sim_config as CFG

# OpenGL imports - only imported if OpenGL rendering is used
try:
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
    import ctypes
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Using Pygame renderer only.")


def km_to_screen_coords(x_km, y_km, x_range, y_range, screen_width, screen_height):
    """Convert kilometer coordinates to screen pixels"""
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    screen_x = int((x_km - x_range[0]) / range_x * screen_width)
    screen_y = int((y_km - y_range[0]) / range_y * screen_height)
    return screen_x, screen_y


def create_cloud_surface(ellipse_params, domain_size, area_size_km, width, height, x_range, y_range):
    """
    Create a Pygame surface with a cloud drawn on it.
    
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
    
    # Convert to km
    cx_km = cx / domain_size * area_size_km
    cy_km = cy / domain_size * area_size_km
    cw_km = cw / domain_size * area_size_km
    ch_km = ch / domain_size * area_size_km
    
    # Convert to screen coordinates
    cx_px, cy_px = km_to_screen_coords(cx_km, cy_km, x_range, y_range, width, height)
    
    # Calculate size in pixels
    range_x = x_range[1] - x_range[0]
    range_y = y_range[1] - y_range[0]
    cw_px = int(cw_km / range_x * width)
    ch_px = int(ch_km / range_y * height)
    
    # Ensure minimum size
    cw_px = max(10, cw_px)
    ch_px = max(10, ch_px)
    
    # Create surface with alpha
    cloud_surface = pygame.Surface((cw_px + 4, ch_px + 4), pygame.SRCALPHA)
    
    # Choose cloud color based on altitude and type
    if cloud_type == "cirrus":
        base_color = (250, 250, 255)  # Slightly blue-white for high clouds
    elif cloud_type == "cumulonimbus":
        base_color = (240, 240, 245)  # Darker for storm clouds
    else:
        base_color = (245, 245, 250)  # Default fluffy white
    
    # Adjust for altitude
    if altitude > 5.0:
        # High clouds are whiter
        brightness = min(1.0, 0.98 + altitude * 0.004)
        base_color = tuple(min(255, int(c * brightness)) for c in base_color)
    elif altitude < 1.0:
        # Low clouds are slightly darker
        darkness = max(0.92, 1.0 - (1.0 - altitude) * 0.1)
        base_color = tuple(int(c * darkness) for c in base_color)
    
    # Draw cloud ellipse
    ellipse_rect = pygame.Rect(2, 2, cw_px, ch_px)
    
    # Apply opacity
    alpha = int(min(0.95, cop) * 255)
    
    # Create gradient effect
    center = (cw_px // 2 + 2, ch_px // 2 + 2)
    for y in range(cloud_surface.get_height()):
        for x in range(cloud_surface.get_width()):
            # Calculate distance to center (normalized)
            dx = (x - center[0]) / (cw_px / 2 + 2)
            dy = (y - center[1]) / (ch_px / 2 + 2)
            distance = math.sqrt(dx**2 + dy**2)
            
            # Only draw inside the ellipse
            if distance <= 1.0:
                # Gradient from center to edge
                gradient_alpha = int(alpha * (1.0 - distance**2))
                
                # Add a subtle color gradient for depth
                r, g, b = base_color
                # Slightly whiter at the top, darker at bottom
                brightness_offset = dy * 0.1
                r = max(0, min(255, int(r * (1.0 - brightness_offset))))
                g = max(0, min(255, int(g * (1.0 - brightness_offset))))
                b = max(0, min(255, int(b * (1.0 - brightness_offset))))
                
                # Add a subtle noise texture
                noise = np.random.random() * 0.1 - 0.05
                r = max(0, min(255, int(r * (1.0 + noise))))
                g = max(0, min(255, int(g * (1.0 + noise))))
                b = max(0, min(255, int(b * (1.0 + noise))))
                
                cloud_surface.set_at((x, y), (r, g, b, gradient_alpha))
    
    # Calculate position (top-left corner for blitting)
    pos = (cx_px - cw_px // 2, cy_px - ch_px // 2)
    
    # Rotate if needed
    if crot != 0:
        # Convert rotation to degrees
        rot_degrees = math.degrees(crot)
        cloud_surface = pygame.transform.rotate(cloud_surface, rot_degrees)
        
        # Adjust position for rotation
        rot_rect = cloud_surface.get_rect(center=(cx_px, cy_px))
        pos = rot_rect.topleft
    
    return cloud_surface, pos


def draw_vectorized_clouds(screen, cloud_ellipses, domain_size, area_size_km, width, height, x_range, y_range):
    """
    UPGRADED: Rock-solid vectorized cloud rendering with proper numeric/string split and NE→SW head-room.
    
    Args:
        screen: Pygame surface to draw on
        cloud_ellipses: List of cloud ellipse parameters
        domain_size: Size of simulation domain in meters
        area_size_km: Size of simulation area in kilometers
        width, height: Screen dimensions
        x_range, y_range: Coordinate ranges in km
    """
    if not cloud_ellipses:
        return
    
    # --- fast vectorised path -------------------------------------------------
    # 1️⃣ Separate the 7 numeric fields from the 8th string field
    numeric = np.asarray([e[:7] for e in cloud_ellipses], dtype=np.float32)
    ctype_l = [e[7] if len(e) > 7 else "cumulus" for e in cloud_ellipses]  # keep for colour lookup
    
    # 2️⃣ Unpack the numeric columns (NumPy view, no copies)
    cx, cy, w, h, rot, opac, z = numeric.T
    
    # Convert domain coordinates to km
    cx_km = cx / domain_size * area_size_km
    cy_km = cy / domain_size * area_size_km
    cw_km = w / domain_size * area_size_km
    ch_km = h / domain_size * area_size_km
    
    # STEP 2: Optional sanity head-room (helps us see clouds slightly outside the normal box)
    y_min, y_max = y_range[0], y_range[1]
    x_min, x_max = x_range[0], x_range[1]
    
    # Insert a small ±10% buffer for y-mapping
    pad_y = 0.10 * (y_max - y_min)
    y_min -= pad_y
    y_max += pad_y
    range_y = y_max - y_min
    
    # Same for x-mapping
    pad_x = 0.10 * (x_max - x_min)
    x_min -= pad_x
    x_max += pad_x
    range_x = x_max - x_min
    
    # Convert to screen coordinates (vectorized)
    cx_px = ((cx_km - x_min) / range_x * width).astype(np.int32)
    cy_px = ((cy_km - y_min) / range_y * height).astype(np.int32)
    cw_px = np.maximum(10, (cw_km / range_x * width).astype(np.int32))
    ch_px = np.maximum(10, (ch_km / range_y * height).astype(np.int32))
    
    # Render each cloud using vectorized data
    for i in range(len(numeric)):
        # Extract individual cloud parameters
        cloud_type = ctype_l[i]
        opacity = opac[i]
        altitude = z[i]
        rotation = rot[i]
        
        center_x = int(cx_px[i])
        center_y = int(cy_px[i])
        width_px = int(cw_px[i])
        height_px = int(ch_px[i])
        
        # Skip clouds that are completely off-screen (even with head-room)
        if (center_x < -width_px or center_x > width + width_px or
            center_y < -height_px or center_y > height + height_px):
            continue
        
        # Create cloud surface
        cloud_surface = pygame.Surface((width_px + 4, height_px + 4), pygame.SRCALPHA)
        
        # Choose cloud color based on type
        if cloud_type == "cirrus":
            base_color = (250, 250, 255)
        elif cloud_type == "cumulonimbus":
            base_color = (240, 240, 245)
        else:
            base_color = (245, 245, 250)
        
        # Adjust for altitude
        if altitude > 5.0:
            brightness = min(1.0, 0.98 + altitude * 0.004)
            base_color = tuple(min(255, int(c * brightness)) for c in base_color)
        elif altitude < 1.0:
            darkness = max(0.92, 1.0 - (1.0 - altitude) * 0.1)
            base_color = tuple(int(c * darkness) for c in base_color)
        
        # Apply opacity
        alpha = int(min(0.95, opacity) * 255)
        
        # Fast ellipse drawing using pygame primitives
        ellipse_rect = pygame.Rect(2, 2, width_px, height_px)
        
        # Create a simple gradient ellipse
        center = (width_px // 2 + 2, height_px // 2 + 2)
        
        # Use pygame's built-in ellipse drawing for speed
        # Draw multiple concentric ellipses for gradient effect
        for layer in range(5):
            layer_alpha = alpha * (1.0 - layer * 0.15)
            if layer_alpha <= 0:
                break
            
            layer_color = tuple(int(c * (1.0 - layer * 0.05)) for c in base_color)
            layer_surface = pygame.Surface((width_px + 4, height_px + 4), pygame.SRCALPHA)
            
            # Shrink ellipse for each layer
            shrink = layer * 2
            layer_rect = pygame.Rect(
                2 + shrink, 2 + shrink, 
                max(1, width_px - shrink * 2), 
                max(1, height_px - shrink * 2)
            )
            
            pygame.draw.ellipse(layer_surface, (*layer_color, int(layer_alpha)), layer_rect)
            cloud_surface.blit(layer_surface, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
        
        # Calculate position for blitting
        pos_x = center_x - width_px // 2
        pos_y = center_y - height_px // 2
        
        # Apply rotation if needed
        if abs(rotation) > 0.01:
            rot_degrees = math.degrees(rotation)
            cloud_surface = pygame.transform.rotate(cloud_surface, rot_degrees)
            
            # Adjust position for rotation
            rot_rect = cloud_surface.get_rect(center=(center_x, center_y))
            pos_x, pos_y = rot_rect.topleft
        
        # Blit cloud to screen
        screen.blit(cloud_surface, (pos_x, pos_y))
    
    # STEP 4: Optional FPS monitor
    frame_idx = getattr(draw_vectorized_clouds, '_frame_counter', 0)
    draw_vectorized_clouds._frame_counter = frame_idx + 1
    
    if getattr(CFG, 'DEBUG', False) and frame_idx % 30 == 0:
        print(f"[RENDER] drew {len(cloud_ellipses)} ellipses in vector mode")


def _draw_clouds_fallback(screen, cloud_ellipses, domain_size, area_size_km, width, height, x_range, y_range):
    """
    STEP 3: Fallback cloud rendering method (commented out in normal use).
    """
    # This fallback is now optional and rarely used since vectorized path is rock-solid
    print("Using fallback cloud rendering... (this should rarely happen)")
    for ellipse_params in cloud_ellipses:
        try:
            cloud_surface, pos = create_cloud_surface(
                ellipse_params, domain_size, area_size_km, 
                width, height, x_range, y_range
            )
            screen.blit(cloud_surface, pos)
        except Exception as e:
            print(f"Failed to render individual cloud: {e}")
            continue


def draw_cloud_trail(screen, cloud_positions, x_range, y_range, width, height):
    """
    Draw a trail showing recent cloud movement.
    
    Args:
        screen: Pygame surface to draw on
        cloud_positions: List of (x_km, y_km) tuples
        x_range, y_range: Coordinate ranges in km
        width, height: Screen dimensions
    """
    if not cloud_positions or len(cloud_positions) < 2:
        return
    
    # Convert all positions to screen coordinates
    screen_positions = [
        km_to_screen_coords(x, y, x_range, y_range, width, height)
        for x, y in cloud_positions
    ]
    
    # Draw trail with increasing transparency
    for i in range(len(screen_positions) - 1):
        start_pos = screen_positions[i]
        end_pos = screen_positions[i + 1]
        
        # Calculate alpha based on position in history
        alpha = int(255 * (i / len(screen_positions)))
        
        # Draw line segment
        pygame.draw.line(
            screen,
            (200, 200, 200, alpha),
            start_pos,
            end_pos,
            width=2
        )


def draw_clouds_batch(screen, cloud_ellipses, domain_size, area_size_km, width, height, x_range, y_range):
    """
    Optimized batch cloud drawing function that automatically chooses the best rendering method.
    
    Args:
        screen: Pygame surface to draw on
        cloud_ellipses: List of cloud ellipse parameters
        domain_size: Size of simulation domain in meters
        area_size_km: Size of simulation area in kilometers
        width, height: Screen dimensions
        x_range, y_range: Coordinate ranges in km
    """
    if not cloud_ellipses:
        return
    
    # Use vectorized rendering for multiple clouds
    if len(cloud_ellipses) > 1:
        draw_vectorized_clouds(screen, cloud_ellipses, domain_size, area_size_km, width, height, x_range, y_range)
    else:
        # Use individual rendering for single clouds
        for ellipse_params in cloud_ellipses:
            cloud_surface, pos = create_cloud_surface(
                ellipse_params, domain_size, area_size_km, 
                width, height, x_range, y_range
            )
            screen.blit(cloud_surface, pos)


# ===== OpenGL Rendering Support =====

class GLCloudRenderer:
    """OpenGL-based cloud renderer for smoother cloud effects"""
    
    # Vertex shader - with interpolation support
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texCoords;
    
    out vec2 fragTexCoords;
    
    uniform mat4 projection;
    uniform mat4 model;
    uniform vec2 prevPosition;
    uniform vec2 velocity;
    uniform float u_alpha;
    
    void main() {
        // Interpolate position based on alpha
        vec2 interpolatedPos = prevPosition + u_alpha * velocity;
        
        // Apply model matrix (which contains the interpolated position)
        gl_Position = projection * model * vec4(position, 0.0, 1.0);
        fragTexCoords = texCoords;
    }
    """
    
    # Fragment shader - smooth cloud rendering with cloud type coloring
    FRAGMENT_SHADER = """
    #version 330 core
    in vec2 fragTexCoords;
    
    out vec4 fragColor;
    
    uniform sampler2D cloudTexture;
    uniform float opacity;
    uniform vec4 cloudColor;
    uniform float altitude;
    uniform int cloudType;  // 0=cirrus, 1=cumulus, 2=cumulonimbus
    
    void main() {
        // Calculate distance from center (0.5, 0.5)
        vec2 center = vec2(0.5, 0.5);
        float dist = distance(fragTexCoords, center);
        
        // Cloud shape with soft edges
        float alpha = smoothstep(0.5, 0.3, dist) * opacity;
        
        // Add some noise for texture
        float noise = fract(sin(dot(fragTexCoords, vec2(12.9898, 78.233))) * 43758.5453) * 0.1;
        alpha *= (1.0 - noise);
        
        // Altitude effect - higher clouds are brighter at the edges
        float altitude_factor = min(1.0, altitude / 5.0);
        float edge_highlight = smoothstep(0.35, 0.5, dist) * altitude_factor * 0.3;
        
        // Cloud type effects
        vec3 finalColor = cloudColor.rgb;
        
        // Cirrus clouds are brighter and more transparent
        if (cloudType == 0) {  // Cirrus
            finalColor = vec3(1.0, 1.0, 1.0);
            float wispy = sin(fragTexCoords.x * 20.0) * 0.5 + 0.5;
            alpha *= wispy * 0.7 + 0.3;
        }
        // Cumulus clouds are puffy and white
        else if (cloudType == 1) {  // Cumulus
            finalColor = vec3(0.95, 0.95, 0.95);
        }
        // Cumulonimbus clouds are darker and denser
        else if (cloudType == 2) {  // Cumulonimbus
            finalColor = vec3(0.85, 0.85, 0.9);
            alpha *= 1.2;  // Denser
        }
        
        // Add edge highlight
        finalColor += vec3(edge_highlight);
        
        // Output color
        fragColor = vec4(finalColor, alpha);
    }
    """
    
    def __init__(self, screen_size):
        """Initialize the OpenGL cloud renderer
        
        Args:
            screen_size: Tuple of (width, height) in pixels
        """
        self.screen_size = screen_size
        self.initialized = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.projection_matrix = self._create_ortho_matrix(0, screen_size[0], screen_size[1], 0, -1, 1)
        
        # Cloud texture
        self.cloud_texture = None
    
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
            print(f"Error compiling shaders: {e}")
            return False
        
        # Create a simple quad for each cloud
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
        
        # Create and bind VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create and bind EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Texture coord attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))
        glEnableVertexAttribArray(1)
        
        # Create cloud texture
        self.cloud_texture = self._create_cloud_texture()
        
        # Set OpenGL state
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Unbind VAO
        glBindVertexArray(0)
        
        self.initialized = True
        return True
    
    def _create_cloud_texture(self, size=256):
        """Create a sophisticated cloud texture"""
        # Create a circular gradient texture
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        
        # Generate a radial gradient for the cloud
        data = np.zeros((size, size, 4), dtype=np.uint8)
        center = size // 2
        radius = size // 2 - 2
        
        # Create a perlin-like noise function
        def noise(x, y, scale=10.0):
            # Simple pseudo-perlin noise
            n = x * 12.9898 + y * 78.233
            return np.sin(n) * 43758.5453 % 1
        
        for y in range(size):
            for x in range(size):
                # Calculate normalized coordinates
                nx = (x - center) / radius
                ny = (y - center) / radius
                dist = np.sqrt(nx**2 + ny**2)
                
                # Base cloud shape (circular gradient)
                if dist <= 1.0:
                    # Add some turbulence to the edge
                    noise_scale = 10.0
                    turbulence = noise(x/noise_scale, y/noise_scale) * 0.2
                    
                    # Soft falloff from center
                    alpha = max(0, 1 - (dist + turbulence)**2)
                    
                    # Add fine details with multiple noise layers
                    detail1 = noise(x/5.0, y/5.0) * 0.1
                    detail2 = noise(x/20.0, y/20.0) * 0.2
                    
                    # Combine for final alpha
                    final_alpha = alpha * (1.0 + detail1 - detail2)
                    final_alpha = max(0, min(1, final_alpha))
                    
                    data[y, x, 0] = 255  # R
                    data[y, x, 1] = 255  # G
                    data[y, x, 2] = 255  # B
                    data[y, x, 3] = int(final_alpha * 255)  # A
                else:
                    data[y, x, 3] = 0  # Transparent outside
        
        # Set texture parameters with LINEAR filtering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Upload texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        
        return texture
    
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
    
    def render_clouds(self, cloud_ellipses, x_range, y_range, width, height, alpha=0.0):
        """Render clouds with OpenGL
        
        Args:
            cloud_ellipses: List of cloud ellipse parameters
            x_range, y_range: Coordinate ranges in km
            width, height: Screen dimensions
            alpha: Interpolation factor for smooth movement (0-1)
        """
        if not self.initialized and not self.init_gl():
            print("Failed to initialize OpenGL. Skipping cloud rendering.")
            return
        
        # Prepare for rendering
        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)
        
        # Set projection matrix uniform
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection_matrix)
        
        # Set texture uniform
        tex_loc = glGetUniformLocation(self.shader_program, "cloudTexture")
        glUniform1i(tex_loc, 0)  # Texture unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.cloud_texture)
        
        # Set interpolation alpha uniform
        alpha_loc = glGetUniformLocation(self.shader_program, "u_alpha")
        glUniform1f(alpha_loc, alpha)
        
        # Range conversion factors
        range_x = x_range[1] - x_range[0]
        range_y = y_range[1] - y_range[0]
        
        # Render each cloud
        for ellipse_params in cloud_ellipses:
            if len(ellipse_params) >= 8:
                cx, cy, cw, ch, crot, cop, altitude, cloud_type = ellipse_params[:8]
            elif len(ellipse_params) >= 7:
                cx, cy, cw, ch, crot, cop, altitude = ellipse_params[:7]
                cloud_type = "cumulus"
            else:
                cx, cy, cw, ch, crot, cop = ellipse_params
                altitude = 1.0
                cloud_type = "cumulus"
            
            # Get previous position and velocity if available
            prev_x, prev_y = cx, cy
            vx, vy = 0, 0
            
            # Convert to km
            cx_km = cx / 1000  # Convert from simulation units to km
            cy_km = cy / 1000
            prev_x_km = prev_x / 1000
            prev_y_km = prev_y / 1000
            cw_km = cw / 1000
            ch_km = ch / 1000
            
            # Convert to screen coordinates
            screen_x = int((cx_km - x_range[0]) / range_x * width)
            screen_y = int((cy_km - y_range[0]) / range_y * height)
            prev_screen_x = int((prev_x_km - x_range[0]) / range_x * width)
            prev_screen_y = int((prev_y_km - y_range[0]) / range_y * height)