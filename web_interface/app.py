"""
Flask web application for browsing and plotting CSV files
"""
from flask import Flask, render_template, request, jsonify
import os
import sys

# Add the project root to the path to import calibration_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_interface.file_scanner import scan_directory_tree
from web_interface.csv_plotter import plot_csv_files

app = Flask(__name__)

@app.route('/')
def index():
    """Main page with tree view and plot area"""
    return render_template('index.html')

@app.route('/api/files')
def get_files():
    """API endpoint to get directory tree structure"""
    try:
        # Get the project root directory (parent of web_interface)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tree_data = scan_directory_tree(project_root)
        return jsonify({'success': True, 'data': tree_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plot', methods=['POST'])
def generate_plot():
    """API endpoint to generate plots from selected CSV files"""
    try:
        data = request.get_json()
        file_paths = data.get('files', [])
        show_error_bars = data.get('showErrorBars', True)  # Default to True for backward compatibility
        
        if not file_paths:
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Convert relative paths to absolute paths
        absolute_paths = []
        for file_path in file_paths:
            absolute_path = os.path.join(project_root, file_path)
            if not os.path.exists(absolute_path):
                return jsonify({'success': False, 'error': f'File not found: {file_path}'}), 404
            absolute_paths.append(absolute_path)
        
        result = plot_csv_files(absolute_paths, show_error_bars)
        return jsonify({
            'success': True, 
            'data': result['plot_data'],
            'statistics': result['statistics']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 
