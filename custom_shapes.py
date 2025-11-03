"""
Custom Shape Creation Module
Allows users to create custom target shapes through various methods.
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import base64
import io
import streamlit as st

def create_mathematical_shape(equation, x_range=(-50, 50), num_points=200):
    """
    Create a shape from a mathematical equation.
    
    Parameters:
    -----------
    equation : str
        Mathematical equation (e.g., "x**2", "sin(x/10)", "abs(x)")
    x_range : tuple
        Range of x values
    num_points : int
        Number of points to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with 'x' and 'y' columns representing the shape
    """
    
    try:
        # Create x values
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        # Evaluate the equation safely
        # Allow common mathematical functions
        allowed_names = {
            "x": x,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "pi": np.pi,
            "e": np.e
        }
        
        # Evaluate the equation
        y = eval(equation, {"__builtins__": {}}, allowed_names)
        
        # Remove any infinite or NaN values
        mask = np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) == 0:
            raise ValueError("Equation produced no valid points")
        
        # Normalize to 0-100 range for consistency
        x_norm = (x_clean - x_clean.min()) / (x_clean.max() - x_clean.min()) * 100
        y_norm = (y_clean - y_clean.min()) / (y_clean.max() - y_clean.min()) * 100
        
        return pd.DataFrame({'x': x_norm, 'y': y_norm})
        
    except Exception as e:
        raise ValueError(f"Error evaluating equation: {str(e)}")

def create_parametric_shape(x_equation, y_equation, t_range=(0, 2*np.pi), num_points=200):
    """
    Create a shape from parametric equations.
    
    Parameters:
    -----------
    x_equation : str
        Parametric equation for x (e.g., "cos(t)")
    y_equation : str
        Parametric equation for y (e.g., "sin(t)")
    t_range : tuple
        Range of parameter t
    num_points : int
        Number of points to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with 'x' and 'y' columns representing the shape
    """
    
    try:
        # Create parameter values
        t = np.linspace(t_range[0], t_range[1], num_points)
        
        # Allowed functions and constants
        allowed_names = {
            "t": t,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "pi": np.pi,
            "e": np.e
        }
        
        # Evaluate both equations
        x = eval(x_equation, {"__builtins__": {}}, allowed_names)
        y = eval(y_equation, {"__builtins__": {}}, allowed_names)
        
        # Remove any infinite or NaN values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) == 0:
            raise ValueError("Equations produced no valid points")
        
        # Normalize to 0-100 range
        x_norm = (x_clean - x_clean.min()) / (x_clean.max() - x_clean.min()) * 100
        y_norm = (y_clean - y_clean.min()) / (y_clean.max() - y_clean.min()) * 100
        
        return pd.DataFrame({'x': x_norm, 'y': y_norm})
        
    except Exception as e:
        raise ValueError(f"Error evaluating parametric equations: {str(e)}")

def points_from_image(image_data, threshold=128, max_points=200):
    """
    Extract points from an uploaded image by detecting edges/contours.
    
    Parameters:
    -----------
    image_data : bytes
        Image data from file upload
    threshold : int
        Threshold for edge detection (0-255)
    max_points : int
        Maximum number of points to extract (reduced for better morphing)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with 'x' and 'y' columns representing the shape
    """
    
    try:
        # Open and process the image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize if too large (for performance)
        if image.size[0] > 400 or image.size[1] > 400:
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple edge detection (you could use more sophisticated methods)
        # Find pixels below threshold (assuming dark lines on light background)
        edge_pixels = np.where(img_array < threshold)
        
        if len(edge_pixels[0]) == 0:
            raise ValueError("No edges detected. Try adjusting the threshold or use a clearer image.")
        
        # Get coordinates (flip y to match typical coordinate system)
        y_coords = edge_pixels[0]
        x_coords = edge_pixels[1]
        y_coords = img_array.shape[0] - y_coords  # Flip y-axis
        
        # Smart subsampling to preserve shape structure
        if len(x_coords) > max_points:
            # Use systematic sampling instead of random to preserve shape
            step = len(x_coords) // max_points
            indices = np.arange(0, len(x_coords), step)[:max_points]
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
        
        # Remove duplicate points to avoid overlapping
        points = np.column_stack([x_coords, y_coords])
        unique_points = np.unique(points, axis=0)
        
        # If still too many points after deduplication, thin them out more
        if len(unique_points) > max_points:
            # Use distance-based thinning - keep points that are sufficiently apart
            selected_indices = [0]  # Always keep first point
            min_distance = 2  # Minimum distance between points
            
            for i in range(1, len(unique_points)):
                current_point = unique_points[i]
                distances = np.sqrt(np.sum((unique_points[selected_indices] - current_point)**2, axis=1))
                
                if np.min(distances) > min_distance:
                    selected_indices.append(i)
                    
                if len(selected_indices) >= max_points:
                    break
            
            unique_points = unique_points[selected_indices]
        
        x_coords = unique_points[:, 0]
        y_coords = unique_points[:, 1]
        
        # Final check - ensure we have enough but not too many points
        if len(x_coords) < 20:
            raise ValueError("Too few points detected. Try lowering the threshold or use a clearer image.")
        
        # Normalize to 0-100 range
        x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min()) * 100
        y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min()) * 100
        
        return pd.DataFrame({'x': x_norm, 'y': y_norm})
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def get_custom_shape_examples():
    """Get example equations for users to try."""
    
    return {
        "Mathematical Functions": [
            ("Parabola", "x**2/100"),
            ("Sine Wave", "50 + 30*sin(x/10)"),
            ("Exponential", "exp(x/20)"),
            ("Absolute Value", "abs(x)"),
            ("Cubic", "x**3/1000"),
            ("Square Root", "sqrt(abs(x))"),
        ],
        "Parametric Shapes": [
            ("Circle", "50*cos(t)", "50*sin(t)"),
            ("Heart", "16*cos(t)**3", "13*sin(t) - 5*sin(2*t) - 2*sin(3*t) - sin(4*t)"),
            ("Spiral", "t*cos(t)", "t*sin(t)"),
            ("Figure 8", "sin(t)", "sin(2*t)"),
            ("Flower", "cos(5*t)*cos(t)", "cos(5*t)*sin(t)"),
            ("Butterfly", "sin(t)*(exp(cos(t)) - 2*cos(4*t) - sin(t/12)**5)", "cos(t)*(exp(cos(t)) - 2*cos(4*t) - sin(t/12)**5)"),
        ]
    }

def create_drawing_canvas():
    """
    Create an interactive drawing canvas using Streamlit.
    Note: This is a simplified version. For full drawing functionality,
    you'd need a more sophisticated frontend component.
    """
    
    st.write("### Interactive Drawing Canvas")
    st.info("Drawing canvas would require a custom Streamlit component. For now, use mathematical equations or image upload.")
    
    # Placeholder for future drawing canvas implementation
    # This would require creating a custom Streamlit component with HTML5 Canvas
    
    return None

def validate_equation(equation, variable="x"):
    """
    Validate that an equation is safe to evaluate.
    
    Parameters:
    -----------
    equation : str
        Mathematical equation to validate
    variable : str
        Variable name used in equation
        
    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    
    # Check for potentially dangerous functions/keywords
    dangerous_keywords = [
        "__", "import", "exec", "eval", "open", "file", "input", "raw_input",
        "compile", "reload", "globals", "locals", "vars", "dir", "help"
    ]
    
    equation_lower = equation.lower()
    for keyword in dangerous_keywords:
        if keyword in equation_lower:
            return False, f"Potentially unsafe keyword detected: {keyword}"
    
    # Check for valid characters (letters, numbers, operators, parentheses, common functions)
    allowed_chars = "abcdefghijklmnopqrstuvwxyz0123456789+-*/().,_"
    if not all(c.lower() in allowed_chars for c in equation.replace(" ", "")):
        return False, "Equation contains invalid characters"
    
    # Try to evaluate with a small test range
    try:
        if variable == "x":
            test_x = np.array([0, 1, 2])
            allowed_names = {
                "x": test_x,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "abs": np.abs, "pi": np.pi, "e": np.e
            }
            result = eval(equation, {"__builtins__": {}}, allowed_names)
            
        elif variable == "t":
            test_t = np.array([0, 1, 2])
            allowed_names = {
                "t": test_t,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "abs": np.abs, "pi": np.pi, "e": np.e
            }
            result = eval(equation, {"__builtins__": {}}, allowed_names)
        
        return True, "Equation is valid"
        
    except Exception as e:
        return False, f"Equation evaluation error: {str(e)}"