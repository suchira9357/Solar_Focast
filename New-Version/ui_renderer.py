import pygame
import numpy as np
import math

# OpenGL imports - only imported if OpenGL rendering is used
try:
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
    import ctypes
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Using Pygame renderer only for UI.")

def draw_text(screen, text, position, font_size=16, color=(0, 0, 0), 
             bg_color=None, bold=False, center=False, padding=0):
    """Draw text on the screen with optional background."""
    font = pygame.font.SysFont('Arial', font_size, bold=bold)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    
    if center:
        text_rect.center = position
    else:
        text_rect.topleft = position
        
    if bg_color is not None:
        # Check if bg_color has alpha
        if len(bg_color) == 4:
            # Create a surface with per-pixel alpha
            bg_surface = pygame.Surface((text_rect.width + padding*2, text_rect.height + padding*2), pygame.SRCALPHA)
            bg_surface.fill(bg_color)
            bg_rect = bg_surface.get_rect()
            bg_rect.center = text_rect.center
            screen.blit(bg_surface, (bg_rect.x, bg_rect.y))
        else:
            # Regular background
            bg_rect = text_rect.inflate(padding*2, padding*2)
            pygame.draw.rect(screen, bg_color, bg_rect, border_radius=padding)
            
    screen.blit(text_surface, text_rect)
    return text_rect

def create_info_panel(screen, info_dict, title=None, position=(20, 20), width=200):
    """Create an information panel with multiple lines of text."""
    # Panel settings
    padding = 10
    line_height = 24
    bg_color = (255, 255, 255, 180)  # Semi-transparent white
    border_color = (0, 0, 0)
    
    # Calculate panel dimensions
    num_lines = len(info_dict) + (1 if title else 0)
    panel_height = padding * 2 + line_height * num_lines
    
    # Create panel background
    panel_rect = pygame.Rect(position[0], position[1], width, panel_height)
    
    # Draw panel background with alpha
    bg_surface = pygame.Surface((width, panel_height), pygame.SRCALPHA)
    bg_surface.fill(bg_color)
    screen.blit(bg_surface, position)
    
    # Draw border
    pygame.draw.rect(screen, border_color, panel_rect, width=1, border_radius=3)
    
    # Draw title if provided
    y_offset = position[1] + padding
    if title:
        draw_text(
            screen, title, 
            (position[0] + padding, y_offset),
            font_size=16, bold=True
        )
        y_offset += line_height
    
    # Draw info lines
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        draw_text(
            screen, text, 
            (position[0] + padding, y_offset),
            font_size=14
        )
        y_offset += line_height
    
    return panel_rect

def draw_trajectory_info(screen, cloud_speed, cloud_direction, confidence, position=(20, 220)):
    """Draw cloud trajectory information with direction arrow."""
    if cloud_speed is None or cloud_direction is None:
        text = "Cloud Movement: Not enough data"
        draw_text(screen, text, position, font_size=14, bold=True)
        return
    
    # Create trajectory panel
    panel_width = 200
    panel_height = 160
    panel_rect = pygame.Rect(position[0], position[1], panel_width, panel_height)
    
    # Panel background
    bg_color = (255, 255, 255, 180)
    bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    bg_surface.fill(bg_color)
    screen.blit(bg_surface, position)
    
    # Panel border
    pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1, border_radius=3)
    
    # Title
    title_pos = (position[0] + 10, position[1] + 10)
    title_rect = draw_text(screen, "Cloud Movement", title_pos, font_size=16, bold=True)
    
    # Speed and direction text
    text_pos = (position[0] + 10, title_rect.bottom + 10)
    speed_text = f"Speed: {cloud_speed:.1f} km/h"
    speed_rect = draw_text(screen, speed_text, text_pos, font_size=14)
    
    dir_pos = (position[0] + 10, speed_rect.bottom + 5)
    dir_text = f"Direction: {cloud_direction:.0f}°"
    dir_rect = draw_text(screen, dir_text, dir_pos, font_size=14)
    
    conf_pos = (position[0] + 10, dir_rect.bottom + 5)
    conf_text = f"Confidence: {int(confidence * 100)}%"
    draw_text(screen, conf_text, conf_pos, font_size=14)
    
    # Direction arrow
    arrow_center = (
        position[0] + panel_width // 2,
        position[1] + panel_height - 40
    )
    arrow_length = 30
    
    # Convert direction from degrees to radians (0° is East, 90° is North)
    direction_rad = math.radians(cloud_direction)
    
    # Calculate arrow endpoint
    end_x = arrow_center[0] + arrow_length * math.cos(direction_rad)
    end_y = arrow_center[1] - arrow_length * math.sin(direction_rad)
    
    # Draw compass circle
    pygame.draw.circle(screen, (230, 230, 230), arrow_center, arrow_length + 5, width=0)
    pygame.draw.circle(screen, (0, 0, 0), arrow_center, arrow_length + 5, width=1)
    
    # Draw N/E/S/W indicators
    compass_points = [
        ("N", 0, -1),
        ("E", 1, 0),
        ("S", 0, 1),
        ("W", -1, 0)
    ]
    
    small_font = pygame.font.SysFont('Arial', 10, bold=True)
    for label, dx, dy in compass_points:
        point_x = arrow_center[0] + (arrow_length + 15) * dx
        point_y = arrow_center[1] + (arrow_length + 15) * dy
        text = small_font.render(label, True, (0, 0, 0))
        text_rect = text.get_rect(center=(point_x, point_y))
        screen.blit(text, text_rect)
    
    # Arrow color based on confidence
    if confidence > 0.7:
        arrow_color = (0, 120, 0)  # Green for high confidence
    elif confidence > 0.3:
        arrow_color = (150, 150, 0)  # Yellow for medium confidence
    else:
        arrow_color = (150, 0, 0)  # Red for low confidence
    
    # Draw arrow
    pygame.draw.line(screen, arrow_color, arrow_center, (end_x, end_y), width=3)
    
    # Draw arrowhead
    head_length = 10
    head_width = 6
    
    # Calculate perpendicular direction for arrowhead
    perp_x = math.sin(direction_rad)
    perp_y = math.cos(direction_rad)
    
    # Calculate arrowhead points
    head_point1 = (
        end_x - head_length * math.cos(direction_rad) + head_width * perp_x,
        end_y + head_length * math.sin(direction_rad) + head_width * perp_y
    )
    head_point2 = (
        end_x - head_length * math.cos(direction_rad) - head_width * perp_x,
        end_y + head_length * math.sin(direction_rad) - head_width * perp_y
    )
    
    # Draw arrowhead
    pygame.draw.polygon(screen, arrow_color, [(end_x, end_y), head_point1, head_point2])
    
    return panel_rect

def draw_affected_panels_list(screen, affected_panels, total_panels, power_output, position=(800, 20)):
    """Draw a list of affected panels with power information."""
    if not affected_panels:
        return
    
    # Panel settings
    panel_width = 300
    panel_height = 250
    padding = 10
    line_height = 20
    
    # Create panel
    panel_rect = pygame.Rect(position[0], position[1], panel_width, panel_height)
    
    # Panel background
    bg_color = (255, 255, 255, 180)
    bg_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    bg_surface.fill(bg_color)
    screen.blit(bg_surface, position)
    
    # Panel border
    pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1, border_radius=3)
    
    # Title
    title_pos = (position[0] + padding, position[1] + padding)
    affected_count = len(affected_panels)
    affected_pct = affected_count / total_panels * 100 if total_panels > 0 else 0
    
    title = f"Affected Panels: {affected_count}/{total_panels} ({affected_pct:.1f}%)"
    title_rect = draw_text(screen, title, title_pos, font_size=16, bold=True)
    
    # List header
    header_pos = (position[0] + padding, title_rect.bottom + 10)
    header = "Panel ID      Reduction      Power"
    header_rect = draw_text(screen, header, header_pos, font_size=14)
    
    # Draw separator line
    line_y = header_rect.bottom + 5
    pygame.draw.line(
        screen, (0, 0, 0), 
        (position[0] + 5, line_y), 
        (position[0] + panel_width - 5, line_y),
        width=1
    )
    
    # List affected panels (up to 10)
    y_pos = line_y + 10
    for i, panel_id in enumerate(affected_panels[:10]):
        if panel_id not in power_output:
            continue
            
        power_data = power_output[panel_id]
        baseline = power_data.get('baseline', 0)
        current = power_data.get('final_power', 0)
        
        if baseline > 0:
            reduction = (baseline - current) / baseline * 100
        else:
            reduction = 0
            
        # Text with formatting
        text = f"{panel_id:<10}   {reduction:>6.1f}%     {current:.2f} kW"
        
        # Color based on reduction
        if reduction > 50:
            color = (180, 0, 0)  # Red for high reduction
        elif reduction > 20:
            color = (180, 120, 0)  # Orange for medium
        else:
            color = (0, 120, 0)  # Green for low
            
        draw_text(screen, text, (position[0] + padding, y_pos), font_size=12, color=color)
        y_pos += line_height
        
    # If there are more affected panels than shown
    if len(affected_panels) > 10:
        more_text = f"...and {len(affected_panels) - 10} more panels"
        draw_text(screen, more_text, (position[0] + padding, y_pos + 5), font_size=12, color=(100, 100, 100))
    
    return panel_rect

def draw_time_slider(screen, current_hour, sunrise_hour, sunset_hour, position, width):
    """Draw a time slider showing current simulation time in daylight context."""
    # Settings
    height = 20
    padding = 5
    
    # Convert hours to positions
    day_start = 0
    day_end = 24
    day_length = day_end - day_start
    
    # Calculate pixel positions
    x_pos = position[0]
    y_pos = position[1]
    
    # Draw background bar
    bg_rect = pygame.Rect(x_pos, y_pos, width, height)
    pygame.draw.rect(screen, (220, 220, 220), bg_rect, border_radius=height//2)
    pygame.draw.rect(screen, (0, 0, 0), bg_rect, width=1, border_radius=height//2)
    
    # Draw daylight portion
    if sunrise_hour < sunset_hour:
        daylight_start = (sunrise_hour - day_start) / day_length * width
        daylight_width = (sunset_hour - sunrise_hour) / day_length * width
        
        daylight_rect = pygame.Rect(
            x_pos + daylight_start, 
            y_pos, 
            daylight_width, 
            height
        )
        
        # Gradient from dark blue to yellow to dark blue
        gradient_surface = pygame.Surface((int(daylight_width), height), pygame.SRCALPHA)
        
        # Create gradient
        for x in range(int(daylight_width)):
            # Position in daylight (0 to 1)
            pos = x / daylight_width
            
            # Color varies from dark blue to yellow to dark blue
            if pos < 0.5:
                # Morning: Blue to yellow
                r = int(255 * (pos * 2))
                g = int(200 * (pos * 2))
                b = int(255 * (1 - pos))
            else:
                # Afternoon: Yellow to blue
                adjusted_pos = (pos - 0.5) * 2  # 0 to 1
                r = int(255 * (1 - adjusted_pos))
                g = int(200 * (1 - adjusted_pos))
                b = int(255 * adjusted_pos)
                
            # Draw vertical line of this color
            pygame.draw.line(gradient_surface, (r, g, b), (x, 0), (x, height))
            
        # Apply the gradient
        screen.blit(gradient_surface, (x_pos + daylight_start, y_pos))
        
        # Redraw border (it got covered by our gradient)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect, width=1, border_radius=height//2)
    
    # Draw hour markers
    for hour in range(day_start, day_end + 1, 3):
        marker_x = x_pos + (hour - day_start) / day_length * width
        
        # Taller markers for main hours
        marker_height = height + 5
        
        # Draw marker line
        pygame.draw.line(
            screen, (0, 0, 0),
            (marker_x, y_pos + height),
            (marker_x, y_pos + height + 5),
            width=1
        )
        
        # Draw hour text
        hour_text = f"{hour:02d}:00"
        draw_text(
            screen, hour_text,
            (marker_x, y_pos + height + 8),
            font_size=10, center=True
        )
    
    # Draw current time indicator
    if day_start <= current_hour <= day_end:
        current_x = x_pos + (current_hour - day_start) / day_length * width
        
        # Triangle pointer
        pointer_height = 15
        pointer_width = 10
        
        pointer_points = [
            (current_x, y_pos - 5),
            (current_x - pointer_width//2, y_pos - 5 - pointer_height),
            (current_x + pointer_width//2, y_pos - 5 - pointer_height)
        ]
        
        pygame.draw.polygon(screen, (200, 0, 0), pointer_points)
        
        # Current time text
        hour = int(current_hour)
        minute = int((current_hour - hour) * 60)
        time_text = f"{hour:02d}:{minute:02d}"
        
        draw_text(
            screen, time_text,
            (current_x, y_pos - 25),
            font_size=14, bold=True, center=True,
            bg_color=(255, 255, 255, 200), padding=5
        )


# ===== OpenGL UI Renderer =====

class GLUIRenderer:
    """OpenGL-based renderer for UI elements with advanced visual effects"""
    
    # Vertex shader for UI elements
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texCoords;
    
    out vec2 fragTexCoords;
    
    uniform mat4 projection;
    uniform mat4 model;
    
    void main() {
        gl_Position = projection * model * vec4(position, 0.0, 1.0);
        fragTexCoords = texCoords;
    }
    """
    
    # Fragment shader for UI panels with smooth gradients and rounded corners
    FRAGMENT_SHADER = """
    #version 330 core
    in vec2 fragTexCoords;
    
    out vec4 fragColor;
    
    uniform vec4 panelColor;
    uniform float cornerRadius;  // 0.0 to 0.5
    uniform vec2 panelSize;      // Width and height in pixels
    uniform float borderWidth;   // Border width in pixels
    uniform vec4 borderColor;
    
    void main() {
        // Calculate distance from edges in pixels
        vec2 pixelPos = fragTexCoords * panelSize;
        vec2 edgeDistance = min(pixelPos, panelSize - pixelPos);
        
        // Corner radius in pixels
        float radius = cornerRadius * min(panelSize.x, panelSize.y);
        
        // Distance from nearest corner
        vec2 cornerVector = vec2(
            max(0.0, radius - edgeDistance.x),
            max(0.0, radius - edgeDistance.y)
        );
        float cornerDistance = length(cornerVector);
        
        // Check if inside border
        bool insideBorder = (edgeDistance.x >= borderWidth && edgeDistance.y >= borderWidth) || 
                            (cornerDistance <= radius - borderWidth);
        
        // Check if inside panel
        bool insidePanel = (edgeDistance.x >= 0.0 && edgeDistance.y >= 0.0) && 
                           (cornerDistance <= radius || cornerVector == vec2(0.0));
        
        // Apply colors based on position
        if (!insidePanel) {
            // Outside panel
            fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        } else if (!insideBorder) {
            // In border
            fragColor = borderColor;
        } else {
            // Inside panel
            // Apply a subtle gradient for depth
            float gradientFactor = (fragTexCoords.y * 0.4 + 0.6);
            vec4 adjustedColor = vec4(
                panelColor.rgb * gradientFactor,
                panelColor.a
            );
            fragColor = adjustedColor;
        }
    }
    """
    
    def __init__(self, screen_size):
        """Initialize the OpenGL UI renderer
        
        Args:
            screen_size: Tuple of (width, height) in pixels
        """
        self.screen_size = screen_size
        self.initialized = False
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.projection_matrix = self._create_ortho_matrix(0, screen_size[0], screen_size[1], 0, -1, 1)
    
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
            print(f"Error compiling UI shaders: {e}")
            return False
        
        # Create a quad for UI panels
        vertices = np.array([
            # x, y, u, v
            0.0, 0.0, 0.0, 0.0,  # Bottom left
            1.0, 0.0, 1.0, 0.0,  # Bottom right
            1.0, 1.0, 1.0, 1.0,  # Top right
            0.0, 1.0, 0.0, 1.0   # Top left
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
        
        # Unbind VAO
        glBindVertexArray(0)
        
        self.initialized = True
        return True
    
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
    
    def render_panel(self, position, size, color=(1.0, 1.0, 1.0, 0.8), corner_radius=0.1, border_width=1.0, border_color=(0.0, 0.0, 0.0, 1.0)):
        """Render a UI panel with OpenGL
        
        Args:
            position: Tuple of (x, y) in pixels
            size: Tuple of (width, height) in pixels
            color: RGBA color tuple (0-1 range)
            corner_radius: Corner radius as fraction of panel size (0-0.5)
            border_width: Border width in pixels
            border_color: RGBA border color tuple (0-1 range)
        """
        if not self.initialized and not self.init_gl():
            print("Failed to initialize OpenGL for UI. Skipping rendering.")
            return
        
        # Prepare for rendering
        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)
        
        # Set projection matrix uniform
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection_matrix)
        
        # Set panel properties uniforms
        color_loc = glGetUniformLocation(self.shader_program, "panelColor")
        glUniform4fv(color_loc, 1, color)
        
        radius_loc = glGetUniformLocation(self.shader_program, "cornerRadius")
        glUniform1f(radius_loc, corner_radius)
        
        size_loc = glGetUniformLocation(self.shader_program, "panelSize")
        glUniform2f(size_loc, size[0], size[1])
        
        border_width_loc = glGetUniformLocation(self.shader_program, "borderWidth")
        glUniform1f(border_width_loc, border_width)
        
        border_color_loc = glGetUniformLocation(self.shader_program, "borderColor")
        glUniform4fv(border_color_loc, 1, border_color)
        
        # Create model matrix for this panel
        model_matrix = np.identity(4, dtype=np.float32)
        
        # Translate
        model_matrix[0, 3] = position[0]
        model_matrix[1, 3] = position[1]
        
        # Scale
        model_matrix[0, 0] = size[0]
        model_matrix[1, 1] = size[1]
        
        # Set model matrix uniform
        model_loc = glGetUniformLocation(self.shader_program, "model")
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix)
        
        # Draw panel
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        # Clean up
        glBindVertexArray(0)
        glUseProgram(0)


def initialize_gl_ui_renderer(screen_size):
    """Initialize OpenGL UI renderer
    
    Args:
        screen_size: Tuple of (width, height) in pixels
    
    Returns:
        GLUIRenderer instance or None if initialization failed
    """
    if not OPENGL_AVAILABLE:
        print("OpenGL not available, cannot initialize GL UI renderer")
        return None
        
    try:
        renderer = GLUIRenderer(screen_size)
        if not renderer.init_gl():
            print("Failed to initialize GL UI renderer")
            return None
            
        return renderer
    except Exception as e:
        print(f"Error initializing OpenGL UI renderer: {e}")
        return None


def gl_create_info_panel(ui_renderer, info_dict, title=None, position=(20, 20), width=200):
    """Create an information panel with OpenGL
    
    Args:
        ui_renderer: GLUIRenderer instance
        info_dict: Dictionary of info items to display
        title: Optional title text
        position: Tuple of (x, y) position in pixels
        width: Width of panel in pixels
    
    Returns:
        Tuple of (x, y, width, height) representing panel rect
    """
    if ui_renderer is None:
        print("No GL renderer available, skipping GL UI rendering")
        return (position[0], position[1], width, 0)
        
    # Calculate panel height
    padding = 10
    line_height = 24
    num_lines = len(info_dict) + (1 if title else 0)
    panel_height = padding * 2 + line_height * num_lines
    
    # Render panel
    ui_renderer.render_panel(
        position,
        (width, panel_height),
        color=(1.0, 1.0, 1.0, 0.7),
        corner_radius=0.1,
        border_width=1.0
    )
    
    # In a full implementation, we would render the text here using
    # the text renderer, but that requires a more complex implementation
    # with font texture atlases. For now, we'll skip text rendering in OpenGL.
    
    return (position[0], position[1], width, panel_height)