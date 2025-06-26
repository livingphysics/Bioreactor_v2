# CSV File Browser and Plotter Web Interface

A Flask-based web application for browsing CSV files in your project directory and generating interactive plots with fitting and error bands.

## Features

- **Tree View Browser**: Navigate through your project directory structure
- **CSV File Selection**: Select multiple CSV files for plotting
- **Interactive Plots**: Generate plots using Plotly with zoom, pan, and hover capabilities
- **Fitting and Error Bands**: Automatically fit lines and display both homoskedastic and heteroskedastic error bands
- **Multi-file Support**: Plot multiple CSV files on the same graph

## Installation

1. Install the required dependencies:
```bash
pip install Flask==3.0.2 plotly==5.19.0
```

2. Make sure all other project dependencies are installed:
```bash
pip install -r requirements.txt
```

## Usage

1. **Start the web interface**:
```bash
python run_web_interface.py
```

2. **Open your web browser** and go to: `http://localhost:5001`

3. **Browse files**: 
   - The left sidebar shows a tree view of your project directory
   - Only folders containing CSV files and CSV files themselves are shown
   - Click on folder icons to expand/collapse directories

4. **Select files**:
   - Check the boxes next to CSV files you want to plot
   - You can select multiple files to plot them together

5. **Generate plots**:
   - Click "Plot Selected Files" to generate the plot
   - The plot will show:
     - Data points as markers
     - Fitted line
     - Homoskedastic error band (gold)
     - Heteroskedastic error band (light coral)

6. **Interact with plots**:
   - Zoom in/out using mouse wheel or zoom tools
   - Pan by clicking and dragging
   - Hover over points to see data values
   - Use the legend to show/hide traces

## How It Works

- **File Scanning**: The application recursively scans your project directory and builds a tree structure showing only relevant files and folders
- **CSV Parsing**: For each CSV file, it reads the first column as X-axis and the last column as Y-axis
- **Plotting**: Uses the existing `calibration_utils.py` functions for fitting and error band calculations
- **Interactive Display**: Plots are rendered using Plotly for rich interactivity

## Error Handling

- Invalid CSV files will show error messages on the web page
- Files with non-numeric data in the first/last columns will be flagged
- Missing files or permission errors are handled gracefully

## File Structure

```
web_interface/
├── __init__.py          # Package initialization
├── app.py              # Main Flask application
├── file_scanner.py     # Directory scanning logic
├── csv_plotter.py      # CSV parsing and plotting
├── templates/
│   └── index.html      # Main web page template
└── README.md           # This file
```

## Troubleshooting

- **Import errors**: Make sure you're running from the project root directory
- **Missing dependencies**: Install Flask and Plotly as shown above
- **Permission errors**: The app will skip directories it can't access
- **Port conflicts**: If port 5001 is busy, modify the port in `app.py`
