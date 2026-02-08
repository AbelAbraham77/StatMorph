"""
DataMorph Engine - Wrapper for data-morph-ai library
Provides functionality to morph 2D datasets into various shapes while preserving statistical properties.
"""

import pandas as pd
import numpy as np
from data_morph.morpher import DataMorpher
from data_morph.data.dataset import Dataset
from data_morph.shapes.factory import ShapeFactory
import os
import data_morph
import warnings
warnings.filterwarnings('ignore')


def load_starter_dataset(dataset_name="dino"):
    """
    Load a starter dataset from the data-morph library.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load. Available: 'dino', 'cat', 'dog', 'bunny', 
        'panda', 'sheep', 'gorilla', 'music', 'pi', 'python', 'soccer_ball', 'superdatascience'
    
    Returns:
    --------
    pandas.DataFrame
        Dataset with 'x' and 'y' columns
    """
    available_datasets = [
        'dino', 'cat', 'dog', 'bunny', 'panda', 'sheep', 'gorilla', 
        'music', 'pi', 'python', 'soccer_ball', 'superdatascience'
    ]
    
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset must be one of {available_datasets}")
    
    try:
        starter_path = os.path.join(
            os.path.dirname(data_morph.__file__), 
            'data', 
            'starter_shapes', 
            f'{dataset_name}.csv'
        )
        df = pd.read_csv(starter_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}': {str(e)}")


def morph_data(df, shape="circle", decimal_preserve=1, iterations=10000, enable_animation=True, 
               freeze_for=0, ease_in=False, ease_out=False, dataset_name="dataset"):
    """
    Morph a 2D dataset into a specified shape while preserving statistical properties.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with 'x' and 'y' columns
    shape : str, default="circle"
        Target shape for morphing. Options: "circle", "square", "spiral", "heart", "donut"
    decimal_preserve : int, default=1
        Number of decimal places to preserve for statistical properties
    iterations : int, default=10000
        Number of iterations for the morphing process
    enable_animation : bool, default=True
        Whether to enable animation frames during morphing
    freeze_for : int, default=0
        Number of frames to freeze at the start (hold original shape)
    ease_in : bool, default=False
        Whether to ease in at the start of the animation (smooth acceleration)
    ease_out : bool, default=False
        Whether to ease out at the end of the animation (smooth deceleration)
    dataset_name : str, default="dataset"
        Name for the dataset (used in output file naming)
        
    Returns:
    --------
    tuple
        (morphed_df: pandas.DataFrame, gif_path: str or None)
        Morphed dataset with same statistical properties but different shape
        and path to generated GIF (if animation enabled)
        
    Raises:
    -------
    ValueError
        If input DataFrame doesn't have required columns or contains invalid data
    """
    
    # Validate input DataFrame
    validate_dataframe(df)
    
    try:
        # Ensure data is float type to avoid pandas warnings
        df_float = df.copy()
        df_float['x'] = df_float['x'].astype(float)
        df_float['y'] = df_float['y'].astype(float)
        
        # Create Dataset object
        dataset = Dataset(name=dataset_name, data=df_float)
        
        # Create shape factory and target shape
        shape_factory = ShapeFactory(dataset)
        target_shape = shape_factory.generate_shape(shape)
        
        # Set up output directory
        output_dir = "data_morph/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataMorpher instance with animation settings
        morpher = DataMorpher(
            decimals=decimal_preserve,
            in_notebook=False,
            write_images=enable_animation,  # Enable image writing for GIF
            write_data=False,
            output_dir=output_dir,
            num_frames=50 if enable_animation else 1,
            keep_frames=False,
            forward_only_animation=True
        )
        
        # Perform morphing with advanced easing options
        morphed_df = morpher.morph(
            start_shape=dataset,
            target_shape=target_shape,
            iterations=iterations,
            freeze_for=freeze_for,
            ease_in=ease_in,
            ease_out=ease_out
        )
        
        # Determine GIF path
        gif_path = None
        if enable_animation:
            gif_filename = f"{dataset_name}_to_{shape}.gif"
            gif_path = os.path.join(output_dir, gif_filename)
            if not os.path.exists(gif_path):
                gif_path = None  # GIF wasn't created
        
        return morphed_df, gif_path
        
    except Exception as e:
        raise ValueError(f"Morphing failed: {str(e)}")


def get_available_datasets():
    """
    Get list of available starter datasets.
    
    Returns:
    --------
    list
        List of available dataset names
    """
    return [
        'dino', 'cat', 'dog', 'bunny', 'panda', 'sheep', 'gorilla', 
        'music', 'pi', 'python', 'soccer_ball', 'superdatascience'
    ]


def validate_dataframe(df):
    """
    Validate that the input DataFrame has the required structure for morphing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate
        
    Raises:
    -------
    ValueError
        If DataFrame doesn't meet requirements
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame cannot be None or empty")
    
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("DataFrame must contain 'x' and 'y' columns")
    
    if len(df) < 10:
        raise ValueError("Dataset must contain at least 10 data points")
    
    # Check for numeric data
    if not pd.api.types.is_numeric_dtype(df['x']) or not pd.api.types.is_numeric_dtype(df['y']):
        raise ValueError("'x' and 'y' columns must contain numeric data")
    
    # Check for missing values
    if df['x'].isna().any() or df['y'].isna().any():
        raise ValueError("Data cannot contain missing values")


def get_available_shapes():
    """
    Get list of available target shapes for morphing.
    
    Returns:
    --------
    list
        List of available shape names
    """
    return [
        'bullseye', 'circle', 'club', 'diamond', 'dots', 'down_parab', 
        'figure_eight', 'h_lines', 'heart', 'high_lines', 'left_parab', 
        'rectangle', 'right_parab', 'rings', 'scatter', 'slant_down', 
        'slant_up', 'spade', 'spiral', 'star', 'up_parab', 'v_lines', 
        'wide_lines', 'x'
    ]