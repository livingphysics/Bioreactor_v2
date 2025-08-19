import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import argparse
from datetime import datetime

def monitor_plots(csv_file='bioreactor_data.csv', update_interval=5):
    """
    Monitor bioreactor data with live updating plots.
    
    Args:
        csv_file: Path to the CSV file being written by the bioreactor
        update_interval: How often to check for new data (seconds)
    """
    plt.ion()  # Turn on interactive mode
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Bioreactor Live Monitor', fontsize=16)
    
    # Temperature plot
    temp_lines = []
    temp_labels = ['vial_A_temp', 'vial_B_temp', 'vial_C_temp', 'vial_D_temp', 'ambient_temp']
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (label, color) in enumerate(zip(temp_labels, colors)):
        line, = ax1.plot([], [], color=color, label=label, linewidth=2)
        temp_lines.append(line)
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Temperature Data')
    ax1.legend()
    ax1.grid(True)
    
    # OD plot
    od_lines_135 = []
    od_lines_180 = []
    vial_names = ['A', 'B', 'C', 'D']
    
    for i, vial in enumerate(vial_names):
        line_135, = ax2.plot([], [], color=colors[i], label=f'Vial {vial} 135°', linewidth=2)
        line_180, = ax2.plot([], [], color=colors[i], linestyle='--', label=f'Vial {vial} 180°', linewidth=2)
        od_lines_135.append(line_135)
        od_lines_180.append(line_180)
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Optical Density (OD)')
    ax2.set_title('Optical Density Data')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    last_size = 0
    times = []
    temp_data = {label: [] for label in temp_labels}
    od_135_data = {f'vial_{vial}_135_degree': [] for vial in vial_names}
    od_180_data = {f'vial_{vial}_180_degree': [] for vial in vial_names}
    
    print(f"Monitoring {csv_file} for live updates...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            if os.path.exists(csv_file):
                current_size = os.path.getsize(csv_file)
                
                if current_size > last_size:
                    # Read the CSV file
                    try:
                        df = pd.read_csv(csv_file)
                        
                        if len(df) > 0:
                            # Convert time to hours
                            times = df['time'].values / 3600
                            
                            # Update temperature data
                            for label in temp_labels:
                                if label in df.columns:
                                    temp_data[label] = df[label].values
                            
                            # Update OD data
                            for vial in vial_names:
                                od_135_col = f'vial_{vial}_135_degree'
                                od_180_col = f'vial_{vial}_180_degree'
                                
                                if od_135_col in df.columns:
                                    od_135_data[od_135_col] = df[od_135_col].values
                                if od_180_col in df.columns:
                                    od_180_data[od_180_col] = df[od_180_col].values
                            
                            # Update temperature plot
                            for i, (label, line) in enumerate(zip(temp_labels, temp_lines)):
                                if len(temp_data[label]) > 0:
                                    line.set_data(times, temp_data[label])
                            
                            # Update OD plot
                            for i, vial in enumerate(vial_names):
                                od_135_col = f'vial_{vial}_135_degree'
                                od_180_col = f'vial_{vial}_180_degree'
                                
                                if len(od_135_data[od_135_col]) > 0:
                                    od_lines_135[i].set_data(times, od_135_data[od_135_col])
                                if len(od_180_data[od_180_col]) > 0:
                                    od_lines_180[i].set_data(times, od_180_data[od_180_col])
                            
                            # Update plot limits
                            if len(times) > 0:
                                ax1.set_xlim(0, max(times))
                                ax2.set_xlim(0, max(times))
                                
                                # Auto-scale y-axes
                                ax1.relim()
                                ax1.autoscale_view()
                                ax2.relim()
                                ax2.autoscale_view()
                            
                            # Update display
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            
                            print(f"Updated at {datetime.now().strftime('%H:%M:%S')} - {len(df)} data points")
                            
                    except Exception as e:
                        print(f"Error reading CSV: {e}")
                
                last_size = current_size
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor bioreactor data with live plots')
    parser.add_argument('--file', '-f', default='bioreactor_data.csv', 
                       help='CSV file to monitor (default: bioreactor_data.csv)')
    parser.add_argument('--interval', '-i', type=float, default=5.0,
                       help='Update interval in seconds (default: 5.0)')
    
    args = parser.parse_args()
    
    print(f"Starting monitor for file: {args.file}")
    print(f"Update interval: {args.interval} seconds")
    
    monitor_plots(csv_file=args.file, update_interval=args.interval)
