"""
CSV plotter module for generating plots with fitting and error bands
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add the project root to the path to import calibration_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration_code.calibration_utils import get_fit_stats, calculate_homoskedastic_band, calculate_heteroskedastic_band

def read_csv_file(file_path: str) -> tuple:
    """
    Read CSV file and return data with column names.
    Returns: (x_data, y_data, x_label, y_label, filename)
    """
    try:
        df = pd.read_csv(file_path)
        
        if len(df.columns) < 2:
            raise ValueError(f"CSV file must have at least 2 columns, found {len(df.columns)}")
        
        # Get first and last columns
        x_col = df.columns[0]
        y_col = df.columns[-1]
        
        # Extract data, removing any rows with NaN values
        data = df[[x_col, y_col]].dropna()
        
        if len(data) == 0:
            raise ValueError("No valid data points found after removing NaN values")
        
        x_data = data[x_col].values
        y_data = data[y_col].values
        
        # Check if data is numeric
        if not (np.issubdtype(x_data.dtype, np.number) and np.issubdtype(y_data.dtype, np.number)):
            raise ValueError("First and last columns must contain numeric data")
        
        filename = os.path.basename(file_path)
        
        return x_data, y_data, x_col, y_col, filename
        
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {str(e)}")

def generate_plot_data(file_paths: list, show_error_bars: bool = True) -> dict:
    """
    Generate plot data for multiple CSV files.
    Returns Plotly-compatible data structure with statistics.
    """
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    fig = go.Figure()
    statistics = []  # List to store statistics for each file
    
    for i, file_path in enumerate(file_paths):
        try:
            x_data, y_data, x_label, y_label, filename = read_csv_file(file_path)
            
            # Get fitting statistics
            gradient, intercept, r2, rmse, model = get_fit_stats(x_data, y_data)
            
            # Store statistics for this file
            file_stats = {
                'filename': filename,
                'gradient': gradient,
                'intercept': intercept,
                'r2': r2,
                'rmse': rmse,
                'num_points': len(x_data)
            }
            statistics.append(file_stats)
            
            # Generate x grid for smooth line
            x_grid = np.linspace(np.min(x_data), np.max(x_data), 200)
            y_grid = model.predict(x_grid.reshape(-1, 1))
            
            # Calculate error bands only if requested
            if show_error_bars:
                band_low, band_high = calculate_heteroskedastic_band(x_data, y_data, model, x_grid)
                low, high = calculate_homoskedastic_band(x_data, y_data, model, x_grid)
            
            color = colors[i % len(colors)]
            
            # Add scatter plot of data points
            fig.add_trace(go.Scatter(
                x=x_data.tolist(),  # Convert numpy array to list
                y=y_data.tolist(),  # Convert numpy array to list
                mode='markers',
                name=f'{filename} (data)',
                marker=dict(color=color, size=6),
                showlegend=True
            ))
            
            # Add fitted line
            fig.add_trace(go.Scatter(
                x=x_grid.tolist(),  # Convert numpy array to list
                y=y_grid.tolist(),  # Convert numpy array to list
                mode='lines',
                name=f'{filename} (fit)',
                line=dict(color=color, width=2),
                showlegend=True
            ))
            
            # Add error bands only if requested
            if show_error_bars:
                # Add homoskedastic error band
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_grid, x_grid[::-1]]).tolist(),  # Convert numpy array to list
                    y=np.concatenate([high, low[::-1]]).tolist(),  # Convert numpy array to list
                    fill='toself',
                    fillcolor=f'rgba(255, 215, 0, 0.25)',  # Gold with transparency
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{filename} (homo band)',
                    showlegend=False
                ))
                
                # Add heteroskedastic error band
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x_grid, x_grid[::-1]]).tolist(),  # Convert numpy array to list
                    y=np.concatenate([band_high, band_low[::-1]]).tolist(),  # Convert numpy array to list
                    fill='toself',
                    fillcolor=f'rgba(240, 128, 128, 0.25)',  # Light coral with transparency
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{filename} (hetero band)',
                    showlegend=False
                ))
            
        except Exception as e:
            # Add error trace to show the problem
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode='markers',
                name=f'{os.path.basename(file_path)} (ERROR: {str(e)})',
                marker=dict(color='red', size=10, symbol='x'),
                showlegend=True
            ))
            
            # Add error statistics
            statistics.append({
                'filename': os.path.basename(file_path),
                'error': str(e),
                'gradient': None,
                'intercept': None,
                'r2': None,
                'rmse': None,
                'num_points': None
            })
    
    # Update layout
    fig.update_layout(
        title='CSV Data Plot with Fitting and Error Bands',
        xaxis_title=x_label if len(file_paths) > 0 else 'X Axis',
        yaxis_title=y_label if len(file_paths) > 0 else 'Y Axis',
        hovermode='closest',
        template='plotly_white'
    )
    
    return {
        'plot_data': fig.to_dict(),
        'statistics': statistics
    }

def plot_csv_files(file_paths: list, show_error_bars: bool = True) -> dict:
    """
    Main function to plot multiple CSV files.
    Returns Plotly figure data and statistics.
    
    Args:
        file_paths: List of file paths to plot
        show_error_bars: Whether to include error bands (homoskedastic and heteroskedastic)
    """
    if not file_paths:
        raise ValueError("No file paths provided")
    
    return generate_plot_data(file_paths, show_error_bars) 
