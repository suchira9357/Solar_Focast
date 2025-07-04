import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import random

# Directory containing the data
data_dir = r"C:\Users\Suchira_Garusinghe\Desktop\Simulation\simulation2\real_data_analyser"

# Read existing files to understand their structure
def read_existing_data():
    data_files = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(file_path)
                # Assuming the column names are Time and Generation
                if len(df.columns) == 2:
                    # Rename columns to ensure consistency
                    df.columns = ['Time', 'Generation']
                    
                    # Extract panel ID from filename
                    panel_id = filename.split('_')[0]
                    if '.' in panel_id:
                        panel_id = panel_id.split('.')[0]  # Handle cases like D.8kW -> D
                    
                    data_files[panel_id] = df
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return data_files

# Get all panels in the simulation
def get_all_panel_ids():
    # Standard set of panel IDs used in your simulation (A through T)
    return [chr(65 + i) for i in range(20)]  # A through T

# Generate similar generation profile WITHOUT cloud effects
def generate_similar_profile(reference_profiles, panel_id):
    # Find a reference profile to use
    if reference_profiles:
        # Pick a random reference profile
        ref_id = random.choice(list(reference_profiles.keys()))
        ref_df = reference_profiles[ref_id]
        
        # Clone the time series
        times = ref_df['Time'].tolist()
        
        # Create a new generation profile with some random variation
        base_factor = 0.8 + random.random() * 0.4  # 0.8 to 1.2
        multiplier = base_factor
        
        # Add random noise to make each profile unique
        noise_factor = 0.05 + random.random() * 0.05  # 5-10% noise
        
        generations = []
        for i, gen in enumerate(ref_df['Generation']):
            # Apply base scaling factor and noise
            new_gen = gen * multiplier * (1 + random.uniform(-noise_factor, noise_factor))
            
            # Ensure generation is non-negative
            new_gen = max(0, new_gen)
            
            generations.append(new_gen)
        
        # Create DataFrame
        new_df = pd.DataFrame({
            'Time': times,
            'Generation': generations
        })
        
        return new_df
    else:
        # No reference data available, create synthetic data
        print("No reference data available. Creating synthetic profile.")
        times = [f"{h:02d}:{m:02d}" for h in range(6, 20) for m in range(0, 60, 10)]
        
        # Create a bell curve pattern typical of solar generation
        generations = []
        for time_str in times:
            h, m = map(int, time_str.split(':'))
            decimal_time = h + m/60
            
            # Bell curve centered at noon (hour 12)
            hour_factor = max(0, 1 - ((decimal_time - 12) / 6)**2)
            
            # Base generation with random variation
            base_gen = 5000 * hour_factor  # 5 kW peak
            variation = random.uniform(0.9, 1.1)
            
            generations.append(base_gen * variation)
        
        new_df = pd.DataFrame({
            'Time': times,
            'Generation': generations
        })
        
        return new_df

# Create generation data for all panels
def create_all_panel_data():
    # Read existing data
    existing_data = read_existing_data()
    print(f"Found existing data for panels: {list(existing_data.keys())}")
    
    all_panels = get_all_panel_ids()
    
    # Identify which panels need generated data
    missing_panels = [p for p in all_panels if p not in existing_data]
    print(f"Generating data for panels: {missing_panels}")
    
    # Generate and save data for missing panels
    for panel_id in missing_panels:
        # Generate clean data without cloud effects
        panel_data = generate_similar_profile(existing_data, panel_id)
        
        # Save to CSV
        output_file = os.path.join(data_dir, f"{panel_id}_generation.csv")
        panel_data.to_csv(output_file, index=False)
        print(f"Generated and saved data for panel {panel_id} to {output_file}")
    
    # Return a dictionary with all panel data
    result = existing_data.copy()
    for panel_id in missing_panels:
        file_path = os.path.join(data_dir, f"{panel_id}_generation.csv")
        result[panel_id] = pd.read_csv(file_path)
    
    return result

# Plot power generation data to visualize
def plot_panel_data(panel_data, output_file=None):
    plt.figure(figsize=(12, 8))
    
    for panel_id, df in panel_data.items():
        # Convert time strings to plottable format if needed
        if isinstance(df['Time'].iloc[0], str):
            times = range(len(df['Time']))
        else:
            times = df['Time']
            
        plt.plot(times, df['Generation'], label=f"Panel {panel_id}")
    
    plt.xlabel('Time Step')
    plt.ylabel('Power Generation (W)')
    plt.title('Solar Panel Power Generation (Without Cloud Effects)')
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

# Create a combined CSV file for analysis
def create_combined_dataset(panel_data, output_file):
    # Find common time steps across all panels
    common_times = None
    
    for panel_id, df in panel_data.items():
        times = set(df['Time'])
        if common_times is None:
            common_times = times
        else:
            common_times = common_times.intersection(times)
    
    common_times = sorted(list(common_times))
    
    # Create combined dataframe
    combined_data = {'timestamp': common_times}
    
    for panel_id, df in panel_data.items():
        # Create a time to generation mapping
        time_to_gen = dict(zip(df['Time'], df['Generation']))
        
        # Extract generation for common times
        panel_gen = [time_to_gen.get(t, np.nan) for t in common_times]
        combined_data[f'panel_{panel_id}'] = panel_gen
    
    # Create and save dataframe
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(output_file, index=False)
    print(f"Created combined dataset with {len(combined_df)} time points for {len(panel_data)} panels")
    return combined_df

if __name__ == "__main__":
    # Generate data for all panels (without cloud effects)
    all_panel_data = create_all_panel_data()
    
    # Plot to visualize
    plot_panel_data(all_panel_data, os.path.join(data_dir, 'all_panels_generation.png'))
    
    # Create combined dataset
    combined_file = os.path.join(data_dir, 'combined_generation.csv')
    combined_df = create_combined_dataset(all_panel_data, combined_file)
    
    print("Done! Clean generation patterns for all panels have been generated and saved.")