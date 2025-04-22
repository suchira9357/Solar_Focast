import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_clean_visualization(panels_csv, output_path):
    """
    Create a clean visualization with solar panel locations from CSV
    without any marker labels.
    """
    # Load panel coordinates
    panels_df = pd.read_csv(panels_csv)
    print(f"Loaded {len(panels_df)} panel coordinates from {panels_csv}")
    
    # Create a figure with a light background
    plt.figure(figsize=(14, 12))
    ax = plt.gca()
    ax.set_facecolor('#f0f5e6')  # Light green background
    
    # Draw grid
    grid_columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    grid_rows = list(range(1, 11))
    domain_size = (10000, 10000)
    
    # Draw grid lines
    for i in range(len(grid_columns) + 1):
        x = i * (domain_size[0] / len(grid_columns))
        plt.axvline(x, color='lightgray', linestyle='-', linewidth=0.5)
        
        if i < len(grid_columns):
            plt.text(x + (domain_size[0] / len(grid_columns) / 2), domain_size[1] * 1.01, 
                    grid_columns[i], ha='center', fontsize=12)
    
    for i in range(len(grid_rows) + 1):
        y = i * (domain_size[1] / len(grid_rows))
        plt.axhline(y, color='lightgray', linestyle='-', linewidth=0.5)
        
        if i < len(grid_rows):
            plt.text(-domain_size[0] * 0.01, y + (domain_size[1] / len(grid_rows) / 2), 
                    str(grid_rows[i]), va='center', fontsize=12)
    
    # Draw boundary line
    boundary_x = [0, 1000, 3000, 5000, 7000, 9000, 10000, 10000, 8000, 5000, 2000, 0, 0]
    boundary_y = [0, 0, 1000, 1000, 500, 1000, 2000, 8000, 10000, 9000, 8000, 6000, 0]
    plt.plot(boundary_x, boundary_y, 'r--', linewidth=2, label='District Boundary')
    
    # Add water area
    water_x = [0, 0, 1000, 2000, 2000, 1000, 0]
    water_y = [0, 6000, 8000, 8000, 3000, 1000, 0]
    plt.fill(water_x, water_y, color='lightblue', alpha=0.5)
    
    # Plot solar panel locations as blue dots
    plt.scatter(panels_df['x'], panels_df['y'], c='blue', s=30, alpha=0.8, label='Solar Panels')
    
    # Set plot limits
    plt.xlim(-domain_size[0] * 0.02, domain_size[0] * 1.02)
    plt.ylim(-domain_size[1] * 0.02, domain_size[1] * 1.02)
    
    # Add title and labels
    plt.title('Colombo District Grid System - Solar Panel Locations', fontsize=16)
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    
    # Add count information
    plt.text(domain_size[0] * 0.05, domain_size[1] * 0.05, 
             f"Total Solar Panels: {len(panels_df)}", 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend
    plt.legend(loc='lower right')
    
    # Save the visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved clean visualization to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Path to your CSV file with panel coordinates
    panels_csv = r"C:\Users\Suchira_Garusinghe\Desktop\SolarPrediction\xy_coordinates_markers.csv"
    
    # Output path for the new clean visualization
    output_path = r"C:\Users\Suchira_Garusinghe\Desktop\SolarPrediction\clean_panel_visualization.png"
    
    # Create the visualization
    create_clean_visualization(panels_csv, output_path)