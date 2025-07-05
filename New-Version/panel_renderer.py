import pygame
import numpy as np
import colorsys
import math

# OpenGL imports - only imported if OpenGL rendering is used
try:
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
    import ctypes
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not available. Using Pygame renderer only for panels.")

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


# ===== OpenGL Panel Renderer =====

class GLPanelRenderer:
    """OpenGL-based renderer for solar panels with advanced visual effects"""
    
    # Vertex shader for panels
    VERTEX_SHADER = """
    #version 330 core
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texCoords;
    
    out vec2 fragTexCoords;
    out vec2 panelCoord;
    
    uniform mat4 projection;
    uniform mat4 model;
    
    void main() {
        gl_Position = projection * model * vec4(position, 0.0, 1.0);
        fragTexCoords = texCoords;
        panelCoord = position;
    }
    """
    
    # Fragment shader for panels with power-based coloring and shadow effects
    FRAGMENT_SHADER = """
    #version 330 core
    in vec2 fragTexCoords;
    in vec2 panelCoord;
    
    out vec4 fragColor;
    
    uniform float powerPercentage;  // 0.0 to 1.0
    uniform float coverage;         // 0.0 to 1.0
    uniform vec2 highlightCenter;   // -0.5 to 0.5 for reflection highlight
    uniform float time;             // For animations
    uniform bool isHighlighted;     // For selected panel
    
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
        
        // Add subtle highlight based on angle (simulates reflective surface)
        float highlightDistance = distance(fragTexCoords, highlightCenter + vec2(0.5));
        float highlight = smoothstep(0.4, 0.0, highlightDistance) * 0.3;
        baseColor += vec3(highlight);
        
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
        
        // Apply highlight effect for selected panel
        if (isHighlighted) {
            float pulseEffect = (sin(time * 5.0) + 1.0) * 0.5;  // 0 to 1 pulsing
            
            if (isBorder) {
                // Bright pulsing border
                baseColor = mix(vec3(1.0, 1.0, 0.4), vec3(0.0, 1.0, 0.0), pulseEffect);
            } else {
                // Subtle inner glow
                baseColor += vec3(0.1, 0.1, 0.0) * pulseEffect;
            }
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
        self.projection_matrix = self._create_ortho_matrix(0, screen_size[0], screen_size[1], 0, -1, 1)
        self.highlighted_panel = None
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
    
    def render_panels(self, panel_df, panel_coverage, power_output, x_range, y_range, width, height):
        """Render solar panels with OpenGL
        
        Args:
            panel_df: DataFrame with panel information
            panel_coverage: Dictionary mapping panel_id to coverage (0-1)
            power_output: Dictionary mapping panel_id to power output information
            x_range, y_range: Coordinate ranges in km
            width, height: Screen dimensions
        
        Returns:
            List of affected panel IDs sorted by coverage
        """
        if not self.initialized and not self.init_gl():
            print("Failed to initialize OpenGL for panels. Skipping rendering.")
            return []
        
        # Prepare for rendering
        glUseProgram(self.shader_program)
        glBindVertexArray(self.vao)
        
        # Set projection matrix uniform
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, self.projection_matrix)
        
        # Set time uniform for animations
        current_time = pygame.time.get_ticks() / 1000.0 - self.start_time
        time_loc = glGetUniformLocation(self.shader_program, "time")
        glUniform1f(time_loc, current_time)
        
        # Range conversion factors
        range_x = x_range[1] - x_range[0]
        range_y = y_range[1] - y_range[0]
        panel_size_km = 0.4  # Default panel size
        panel_size_px = int(panel_size_km / range_x * width)
        
        affected_panels = []
        
        # Render each panel
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
            
            # Set uniforms for this panel
            power_loc = glGetUniformLocation(self.shader_program, "powerPercentage")
            glUniform1f(power_loc, power_pct)
            
            coverage_loc = glGetUniformLocation(self.shader_program, "coverage")
            glUniform1f(coverage_loc, coverage)
            
            # Random highlight center for each panel (simulates different reflection angles)
            highlight_x = math.sin(x_px * 0.01 + current_time * 0.1) * 0.3
            highlight_y = math.cos(y_px * 0.01 + current_time * 0.2) * 0.3
            highlight_loc = glGetUniformLocation(self.shader_program, "highlightCenter")
            glUniform2f(highlight_loc, highlight_x, highlight_y)
            
            # Check if this panel is highlighted
            is_highlighted = (panel_id == self.highlighted_panel)
            highlight_loc = glGetUniformLocation(self.shader_program, "isHighlighted")
            glUniform1i(highlight_loc, int(is_highlighted))
            
            # Create model matrix for this panel
            model_matrix = np.identity(4, dtype=np.float32)
            
            # Translate
            model_matrix[0, 3] = x_px
            model_matrix[1, 3] = y_px
            
            # Scale
            model_matrix[0, 0] = panel_size_px
            model_matrix[1, 1] = panel_size_px
            
            # Set model matrix uniform
            model_loc = glGetUniformLocation(self.shader_program, "model")
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_matrix)
            
            # Draw panel
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
            
            # Draw panel ID if we don't have too many panels
            # Note: We'd need to use a texture-based approach for text in OpenGL
            # For simplicity, we'll skip panel IDs in the OpenGL renderer
        
        # Clean up
        glBindVertexArray(0)
        glUseProgram(0)
        
        # Sort affected panels by coverage
        affected_panels.sort(key=lambda x: x[1], reverse=True)
        return [panel_id for panel_id, _ in affected_panels]
    
    def set_highlighted_panel(self, panel_id):
        """Set which panel should be highlighted"""
        self.highlighted_panel = panel_id


def gl_draw_solar_panels(screen, panel_renderer, panel_df, panel_coverage, power_output, x_range, y_range, width, height):
    """OpenGL wrapper for drawing solar panels
    
    Args:
        screen: Pygame screen surface
        panel_renderer: GLPanelRenderer instance
        panel_df: DataFrame with panel information
        panel_coverage: Dictionary mapping panel_id to coverage (0-1)
        power_output: Dictionary mapping panel_id to power output information
        x_range, y_range: Coordinate ranges in km
        width, height: Screen dimensions
    
    Returns:
        List of affected panel IDs sorted by coverage
    """
    if panel_renderer is None:
        print("No GL renderer available, skipping GL panel rendering")
        return []
        
    return panel_renderer.render_panels(panel_df, panel_coverage, power_output, x_range, y_range, width, height)


def initialize_gl_panel_renderer(screen_size):
    """Initialize OpenGL panel renderer
    
    Args:
        screen_size: Tuple of (width, height) in pixels
    
    Returns:
        GLPanelRenderer instance or None if initialization failed
    """
    if not OPENGL_AVAILABLE:
        print("OpenGL not available, cannot initialize GL panel renderer")
        return None
        
    try:
        renderer = GLPanelRenderer(screen_size)
        if not renderer.init_gl():
            print("Failed to initialize GL panel renderer")
            return None
            
        return renderer
    except Exception as e:
        print(f"Error initializing OpenGL panel renderer: {e}")
        return None