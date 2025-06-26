"""
File scanner module for building directory tree structure
"""
import os
from typing import List, Dict, Any

def has_csv_files(directory: str) -> bool:
    """Check if a directory contains any CSV files (recursively)"""
    for root, dirs, files in os.walk(directory):
        if any(file.endswith('.csv') for file in files):
            return True
    return False

def scan_directory_tree(root_path: str) -> List[Dict[str, Any]]:
    """
    Recursively scan directory and build tree structure.
    Only includes folders and CSV files. Folders without CSV files are excluded.
    """
    tree_data = []
    
    try:
        items = os.listdir(root_path)
        items.sort()  # Sort alphabetically
        
        for item in items:
            item_path = os.path.join(root_path, item)
            
            # Skip hidden files and the web_interface directory itself
            if item.startswith('.') or item == 'web_interface':
                continue
                
            if os.path.isdir(item_path):
                # Check if directory contains CSV files
                if has_csv_files(item_path):
                    # Recursively scan subdirectory
                    children = scan_directory_tree(item_path)
                    if children:  # Only add if there are children
                        tree_data.append({
                            'name': item,
                            'type': 'directory',
                            'path': os.path.relpath(item_path, os.path.dirname(root_path)),
                            'children': children
                        })
            
            elif item.endswith('.csv'):
                # Add CSV file
                tree_data.append({
                    'name': item,
                    'type': 'file',
                    'path': os.path.relpath(item_path, os.path.dirname(root_path)),
                    'size': os.path.getsize(item_path)
                })
    
    except PermissionError:
        # Skip directories we don't have permission to access
        pass
    except Exception as e:
        print(f"Error scanning {root_path}: {e}")
    
    return tree_data 
