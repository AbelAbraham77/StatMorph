"""
StatMorph: Statistically Preserved Data Augmentation with Interactive ML Evaluation
A Streamlit dashboard for exploring data morphing and its impact on statistical properties and ML performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import warnings
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Import custom modules
from morph_engine import morph_data, validate_dataframe, load_starter_dataset, get_available_datasets, get_available_shapes
from evaluator import evaluate_stats, evaluate_ml, evaluate_ml_comprehensive, evaluate_clustering_performance, create_ml_comparison_table, create_comprehensive_ml_table, create_neural_network_details_table, calculate_preservation_score
from real_world_datasets import get_available_real_world_datasets, load_real_world_dataset, get_dataset_description
from custom_shapes import create_mathematical_shape, create_parametric_shape, get_custom_shape_examples, validate_equation
import time
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="StatMorph",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply consistent styling
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    """Main application function."""
    
    # Title and description
    st.title("📊 StatMorph")
    st.markdown("""
    **Statistically Preserved Data Augmentation with Interactive ML Evaluation**
    
    Transform your 2D datasets into various shapes while preserving key statistical properties. 
    Analyze the impact on data integrity and machine learning performance.
    """)
    
    # Add information about available features
    with st.expander("About StatMorph & Available Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Available Target Shapes:**
            - **Geometric**: circle, rectangle, diamond, star
            - **Mathematical**: spiral, figure_eight, bullseye
            - **Symbolic**: heart, club, spade, x
            - **Linear**: h_lines, v_lines, wide_lines, high_lines
            - **Parabolic**: up_parab, down_parab, left_parab, right_parab
            - **Other**: dots, rings, scatter, slant_up, slant_down
            """)
        
        with col2:
            st.markdown("""
            **Animation Features:**
            - **Freeze Frames**: Hold original shape before morphing
            - **Ease In/Out**: Smooth acceleration/deceleration
            - **Progressive Animation**: Watch points move step by step
            
            **Statistical Preservation:**
            - Mean values (X & Y coordinates)
            - Standard deviation
            - Correlation coefficient
            - Configurable precision levels
            """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV", "Use Starter Dataset", "Real-World Datasets", "Custom Shape Creator"]
    )
    
    # Initialize session state
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'df_morphed' not in st.session_state:
        st.session_state.df_morphed = None
    if 'gif_path' not in st.session_state:
        st.session_state.gif_path = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'current_dataset_name' not in st.session_state:
        st.session_state.current_dataset_name = None
    
    # Data loading section
    df_original = load_data(data_source)
    
    # Update session state only when new data is loaded
    if df_original is not None:
        st.session_state.df_original = df_original
        st.session_state.dataset_loaded = True
        # Reset morphed data when new dataset is loaded
        st.session_state.df_morphed = None
        st.session_state.gif_path = None
    
    # Use data from session state if available
    if st.session_state.df_original is not None and st.session_state.dataset_loaded:
        
        # Show current dataset info
        st.sidebar.success(f"Current Dataset: {st.session_state.current_dataset_name or 'Uploaded CSV'}")
        st.sidebar.info(f"Data Points: {len(st.session_state.df_original)}")
        
        # Add a button to clear/reset the dataset
        if st.sidebar.button("Clear Dataset", key="clear_dataset"):
            st.session_state.df_original = None
            st.session_state.df_morphed = None
            st.session_state.gif_path = None
            st.session_state.dataset_loaded = False
            st.session_state.current_dataset_name = None
            st.rerun()
        
        # Morphing parameters - only show if dataset is loaded
        if st.session_state.dataset_loaded and st.session_state.df_original is not None:
            st.sidebar.header("Morphing Parameters")
            
            shape = st.sidebar.selectbox(
                "Target Shape:",
                get_available_shapes(),
                index=0
            )
            
            decimal_preserve = st.sidebar.slider(
                "Decimal Precision:",
                min_value=1,
                max_value=3,
                value=1,
                help="Lower values allow more shape transformation (1=most flexible, 3=most precise)"
            )
            
            iterations = st.sidebar.slider(
                "Iterations:",
                min_value=5000,
                max_value=20000,
                value=10000,
                step=1000,
                help="More iterations = better shape quality (takes longer)"
            )
            
            enable_animation = st.sidebar.checkbox(
                "Enable Animation",
                value=True,
                help="Show visualization of points moving during morphing"
            )
            
            # Advanced Animation Controls
            with st.sidebar.expander("Advanced Animation Options"):
                freeze_frames = st.slider(
                    "Freeze Frames", 
                    min_value=0, 
                    max_value=50, 
                    value=0,
                    help="Number of frames to hold the original shape before morphing starts"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    ease_in = st.checkbox(
                        "Ease In", 
                        value=False,
                        help="Smooth acceleration at start"
                    )
                with col2:
                    ease_out = st.checkbox(
                        "Ease Out", 
                        value=False,
                        help="Smooth deceleration at end"
                    )
                
                if ease_in or ease_out:
                    st.info("Easing creates smoother, more natural-looking animations!")
                
                if freeze_frames > 0:
                    st.info(f"Animation will hold original shape for {freeze_frames} frames")
            
            # Add helpful information
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Suggestions:**")
            st.sidebar.markdown("• **Precision**: Lower = more visual change")
            st.sidebar.markdown("• **Iterations**: More = better shape quality")
            
            # Morph button
            if st.sidebar.button("Morph Data", type="primary", key="morph_data_btn"):
                with st.spinner("Morphing data... This may take a moment."):
                    try:
                        # Get dataset name for GIF generation
                        dataset_name = st.session_state.current_dataset_name or "uploaded"
                        
                        result = morph_data(
                            st.session_state.df_original,  # Use session state data
                            shape=shape, 
                            decimal_preserve=decimal_preserve, 
                            iterations=iterations,
                            enable_animation=enable_animation,
                            freeze_for=freeze_frames,
                            ease_in=ease_in,
                            ease_out=ease_out,
                            dataset_name=dataset_name
                        )
                        
                        # Handle the result (could be tuple or just DataFrame for backward compatibility)
                        if isinstance(result, tuple):
                            df_morphed, gif_path = result
                            st.session_state.gif_path = gif_path
                        else:
                            df_morphed = result
                            st.session_state.gif_path = None
                        
                        st.session_state.df_morphed = df_morphed
                        st.sidebar.success("Data morphed successfully!")
                        
                        if st.session_state.gif_path:
                            st.sidebar.success("Animation GIF generated!")
                        
                    except Exception as e:
                        st.sidebar.error(f"Morphing failed: {str(e)}")
            
            # Display results if morphed data exists
            if st.session_state.df_morphed is not None:
                display_results(st.session_state.df_original, st.session_state.df_morphed, shape, st.session_state.gif_path)
                
                # Advanced Features Section
                st.markdown("---")
                with st.expander("Advanced Features & Export Options"):
                    st.subheader("Shape Quality Assessment")
                    if shape in ['circle']:
                        # Circle-specific quality metrics
                        center_x = st.session_state.df_morphed['x'].mean()
                        center_y = st.session_state.df_morphed['y'].mean()
                        distances = np.sqrt((st.session_state.df_morphed['x'] - center_x)**2 + 
                                          (st.session_state.df_morphed['y'] - center_y)**2)
                        cv_distance = distances.std() / distances.mean()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Circle Quality", f"{(1-cv_distance)*100:.1f}%", 
                                    help="Higher is better (100% = perfect circle)")
                        with col2:
                            st.metric("Average Radius", f"{distances.mean():.2f}")
                        with col3:
                            st.metric("Radius Variation", f"{distances.std():.2f}")
                    else:
                        # General shape quality - point movement analysis
                        movement_x = abs(st.session_state.df_original['x'] - st.session_state.df_morphed['x'])
                        movement_y = abs(st.session_state.df_original['y'] - st.session_state.df_morphed['y'])
                        total_movement = np.sqrt(movement_x**2 + movement_y**2)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Max Point Movement", f"{total_movement.max():.2f}")
                        with col2:
                            st.metric("Average Movement", f"{total_movement.mean():.2f}")
                        with col3:
                            st.metric("Movement Variation", f"{total_movement.std():.2f}")
            else:
                # Show original data preview
                st.header("Data Preview")
                display_data_preview(st.session_state.df_original)  # Use session state data
        
        else:
            st.info("Please load a dataset first to begin morphing!")


def load_data(data_source):
    """Load data based on user selection."""
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="CSV file must contain 'x' and 'y' columns"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                validate_dataframe(df)  # This will raise an exception if invalid
                st.sidebar.success(f"Loaded {len(df)} data points")
                return df
                    
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
                return None
    
    elif data_source == "Use Starter Dataset":
        st.sidebar.subheader("Starter Datasets")
        
        dataset_name = st.sidebar.selectbox(
            "Choose a dataset:",
            get_available_datasets(),
            index=0,
            help="Select from built-in datasets from the data-morph library"
        )
        
        # Store dataset name in session state to track changes
        if 'selected_dataset' not in st.session_state:
            st.session_state.selected_dataset = None
        
        # Check if dataset selection changed
        if st.session_state.selected_dataset != dataset_name:
            st.session_state.selected_dataset = dataset_name
            st.session_state.dataset_loaded = False  # Reset loaded state
            st.session_state.df_original = None
            st.session_state.df_morphed = None
        
        if st.sidebar.button("Load Dataset", key="load_dataset_btn") or (st.session_state.selected_dataset == dataset_name and not st.session_state.dataset_loaded):
            try:
                df = load_starter_dataset(dataset_name)
                st.sidebar.success(f"Loaded {dataset_name} dataset with {len(df)} data points")
                st.session_state.current_dataset_name = dataset_name
                return df
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {str(e)}")
                return None
    
    elif data_source == "Real-World Datasets":
        st.sidebar.subheader("Real-World Datasets")
        
        real_world_datasets = get_available_real_world_datasets()
        selected_real_dataset = st.sidebar.selectbox(
            "Choose a real-world dataset:",
            real_world_datasets,
            index=0,
            help="Select from curated real-world datasets"
        )
        
        # Show dataset description
        description = get_dataset_description(selected_real_dataset)
        st.sidebar.info(f"📊 {description}")
        
        # Store dataset name in session state to track changes
        if 'selected_real_dataset' not in st.session_state:
            st.session_state.selected_real_dataset = None
        
        # Check if dataset selection changed
        if st.session_state.selected_real_dataset != selected_real_dataset:
            st.session_state.selected_real_dataset = selected_real_dataset
            st.session_state.dataset_loaded = False
            st.session_state.df_original = None
            st.session_state.df_morphed = None
        
        if st.sidebar.button("Load Real-World Dataset", key="load_real_dataset_btn") or (st.session_state.selected_real_dataset == selected_real_dataset and not st.session_state.dataset_loaded):
            try:
                df = load_real_world_dataset(selected_real_dataset)
                st.sidebar.success(f"Loaded {selected_real_dataset} with {len(df)} data points")
                st.session_state.current_dataset_name = selected_real_dataset
                return df
            except Exception as e:
                st.sidebar.error(f"Error loading dataset: {str(e)}")
                return None
    
    elif data_source == "Custom Shape Creator":
        st.sidebar.subheader("Custom Shape Creator")
        
        shape_method = st.sidebar.radio(
            "Creation method:",
            ["Mathematical Function", "Parametric Equations"]
        )
        
        if shape_method == "Mathematical Function":
            st.sidebar.write("**Create shape from equation: y = f(x)**")
            
            # Show examples
            examples = get_custom_shape_examples()["Mathematical Functions"]
            example_names = [f"{name}: {eq}" for name, eq in examples]
            
            selected_example = st.sidebar.selectbox(
                "Examples:",
                ["Custom"] + example_names,
                help="Select an example or choose 'Custom' to enter your own"
            )
            
            if selected_example == "Custom":
                equation = st.sidebar.text_input(
                    "Enter equation (use 'x' as variable):",
                    placeholder="x**2",
                    help="Examples: x**2, sin(x/10), exp(x/20), abs(x)"
                )
            else:
                equation = selected_example.split(": ")[1]
                st.sidebar.code(f"y = {equation}")
            
            if equation:
                # Validate equation
                is_valid, error_msg = validate_equation(equation, "x")
                if is_valid:
                    st.sidebar.success("✅ Equation is valid")
                    
                    if st.sidebar.button("Generate Shape", key="generate_math_shape"):
                        try:
                            df = create_mathematical_shape(equation)
                            st.sidebar.success(f"Generated shape with {len(df)} points")
                            st.session_state.current_dataset_name = f"Math: y = {equation}"
                            return df
                        except Exception as e:
                            st.sidebar.error(f"Error generating shape: {str(e)}")
                else:
                    st.sidebar.error(f"❌ {error_msg}")
        
        elif shape_method == "Parametric Equations":
            st.sidebar.write("**Create shape from parametric equations**")
            
            # Show examples
            examples = get_custom_shape_examples()["Parametric Shapes"]
            example_names = [f"{name}" for name, _, _ in examples]
            
            selected_param_example = st.sidebar.selectbox(
                "Examples:",
                ["Custom"] + example_names,
                help="Select an example or choose 'Custom'"
            )
            
            if selected_param_example == "Custom":
                x_eq = st.sidebar.text_input("x(t) =", placeholder="cos(t)")
                y_eq = st.sidebar.text_input("y(t) =", placeholder="sin(t)")
            else:
                for name, x_eq, y_eq in examples:
                    if name == selected_param_example:
                        st.sidebar.code(f"x(t) = {x_eq}")
                        st.sidebar.code(f"y(t) = {y_eq}")
                        break
            
            if x_eq and y_eq:
                # Validate equations
                x_valid, x_error = validate_equation(x_eq, "t")
                y_valid, y_error = validate_equation(y_eq, "t")
                
                if x_valid and y_valid:
                    st.sidebar.success("✅ Equations are valid")
                    
                    if st.sidebar.button("Generate Parametric Shape", key="generate_param_shape"):
                        try:
                            df = create_parametric_shape(x_eq, y_eq)
                            st.sidebar.success(f"Generated shape with {len(df)} points")
                            st.session_state.current_dataset_name = f"Parametric: x={x_eq}, y={y_eq}"
                            return df
                        except Exception as e:
                            st.sidebar.error(f"Error generating shape: {str(e)}")
                else:
                    if not x_valid:
                        st.sidebar.error(f"❌ x(t): {x_error}")
                    if not y_valid:
                        st.sidebar.error(f"❌ y(t): {y_error}")
    
    return None


def display_data_preview(df):
    """Display a preview of the original data."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Original Dataset")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['x'], df['y'], alpha=0.6, c='steelblue', s=50)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title('Original Data Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("Data Summary")
        summary_stats = pd.DataFrame({
            'Statistic': ['Count', 'Mean X', 'Mean Y', 'Std X', 'Std Y', 'Correlation'],
            'Value': [
                f"{len(df)}",
                f"{df['x'].mean():.3f}",
                f"{df['y'].mean():.3f}",
                f"{df['x'].std():.3f}",
                f"{df['y'].std():.3f}",
                f"{df['x'].corr(df['y']):.3f}"
            ]
        })
        st.dataframe(summary_stats, hide_index=True)
        
        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), hide_index=True)


def display_results(df_original, df_morphed, shape, gif_path=None):
    """Display the complete analysis results."""
    
    # Header
    st.header("Morphing Results")
    
    # GIF Display Section (if available)
    if gif_path and os.path.exists(gif_path):
        st.subheader("Morphing Animation")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Display the GIF
            with open(gif_path, "rb") as file_:
                contents = file_.read()
                st.image(contents, caption=f"Morphing Animation: {shape.title()}", use_container_width=True)
        
        # Add download button for the GIF
        with col3:
            with open(gif_path, "rb") as file_:
                btn = st.download_button(
                    label="📥 Download GIF",
                    data=file_.read(),
                    file_name=f"morphing_animation_{shape}.gif",
                    mime="image/gif"
                )
        
        st.markdown("---")
    
    # Side-by-side plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Dataset")
        fig1 = create_scatter_plot(df_original, "Original Data", "steelblue")
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        st.subheader(f"Morphed Dataset ({shape.title()})")
        fig2 = create_scatter_plot(df_morphed, f"Morphed Data ({shape.title()})", "coral")
        st.pyplot(fig2)
        plt.close()
    
    # Statistical comparison
    st.header("Statistical Integrity Analysis")
    
    with st.spinner("Calculating statistical differences..."):
        stats_results = evaluate_stats(df_original, df_morphed)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Statistical Comparison")
        st.dataframe(stats_results, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("Key Insights")
        
        # Extract key metrics for summary
        mean_diff_x = float(stats_results[stats_results['Metric'] == 'X Mean Difference']['Value'].iloc[0])
        mean_diff_y = float(stats_results[stats_results['Metric'] == 'Y Mean Difference']['Value'].iloc[0])
        ks_p_x = float(stats_results[stats_results['Metric'] == 'KS Test X (p-value)']['Value'].iloc[0])
        ks_p_y = float(stats_results[stats_results['Metric'] == 'KS Test Y (p-value)']['Value'].iloc[0])
        
        st.metric("Mean Preservation (X)", f"{mean_diff_x:.4f}", 
                 help="Lower values indicate better mean preservation")
        st.metric("Mean Preservation (Y)", f"{mean_diff_y:.4f}",
                 help="Lower values indicate better mean preservation")
        st.metric("Distribution Similarity (X)", f"{ks_p_x:.4f}",
                 help="Higher p-values indicate more similar distributions")
        st.metric("Distribution Similarity (Y)", f"{ks_p_y:.4f}",
                 help="Higher p-values indicate more similar distributions")
    
    # Enhanced ML Performance Analysis
    st.header("Machine Learning Performance Analysis")
    
    # Basic ML Analysis
    st.subheader("Basic ML Performance Comparison")
    with st.spinner("Training basic ML models and comparing performance..."):
        ml_results = evaluate_ml(df_original, df_morphed)
    
    if 'error' not in ml_results:
        ml_comparison = create_ml_comparison_table(ml_results)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(ml_comparison, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Performance Metrics")
            orig_acc = ml_results['original']['accuracy']
            morph_acc = ml_results['morphed']['accuracy']
            acc_diff = ml_results['differences']['accuracy_diff']
            
            st.metric("Original Accuracy", f"{orig_acc:.4f}")
            st.metric("Morphed Accuracy", f"{morph_acc:.4f}")
            st.metric("Accuracy Difference", f"{acc_diff:.4f}",
                     help="Lower values indicate better ML performance preservation")
    else:
        st.error(f"Basic ML evaluation failed: {ml_results['error']}")
    
    # Comprehensive ML Analysis
    with st.expander("Advanced ML Model Comparison", expanded=False):
        st.subheader("Multiple Algorithm Performance Analysis")
        
        with st.spinner("Training multiple ML algorithms for comprehensive comparison..."):
            comprehensive_ml = evaluate_ml_comprehensive(df_original, df_morphed)
        
        if 'error' not in comprehensive_ml:
            comprehensive_table = create_comprehensive_ml_table(comprehensive_ml)
            st.dataframe(comprehensive_table, hide_index=True, use_container_width=True)
            
            # Model performance summary
            st.subheader("Model Performance Summary")
            
            col1, col2, col3 = st.columns(3)
            
            models = comprehensive_ml['original']['models']
            differences = comprehensive_ml['model_differences']
            
            with col1:
                st.write("**Best Performing Models (Original):**")
                orig_accuracies = {name: data['accuracy'] for name, data in models.items() if 'error' not in data}
                if orig_accuracies:
                    best_model = max(orig_accuracies, key=orig_accuracies.get)
                    st.info(f"{best_model}: {orig_accuracies[best_model]:.4f}")
            
            with col2:
                st.write("**Best Performing Models (Morphed):**")
                morph_models = comprehensive_ml['morphed']['models']
                morph_accuracies = {name: data['accuracy'] for name, data in morph_models.items() if 'error' not in data}
                if morph_accuracies:
                    best_morph_model = max(morph_accuracies, key=morph_accuracies.get)
                    st.info(f"{best_morph_model}: {morph_accuracies[best_morph_model]:.4f}")
            
            with col3:
                st.write("**Most Consistent Model:**")
                if differences:
                    acc_diffs = {name: data['accuracy_diff'] for name, data in differences.items()}
                    most_consistent = min(acc_diffs, key=acc_diffs.get)
                    st.success(f"{most_consistent}: Δ{acc_diffs[most_consistent]:.4f}")
            
            # Neural Network Hyperparameter Details
            st.subheader("Neural Network Hyperparameter Tuning Details")
            nn_details = create_neural_network_details_table(comprehensive_ml)
            if not nn_details.empty and 'Error' not in nn_details.columns:
                st.dataframe(nn_details, hide_index=True, use_container_width=True)
                
                # Add explanation
                st.info("""
                **Neural Network Configuration:**
                - **Hidden Layer Sizes**: Architecture of the neural network (number of neurons per layer)
                - **Activation Function**: Non-linear transformation function (ReLU, Tanh)
                - **Alpha**: L2 regularization parameter to prevent overfitting
                - **Learning Rate**: How the model adapts during training
                - **Solver**: Optimization algorithm (Adam, L-BFGS)
                
                The hyperparameters are automatically tuned using GridSearchCV to find the best configuration for each dataset.
                """)
            else:
                st.warning("Neural Network details not available or an error occurred during training.")
        else:
            st.error(f"Comprehensive ML evaluation failed: {comprehensive_ml['error']}")
    
    # Clustering Analysis
    with st.expander("Clustering Performance Analysis", expanded=False):
        st.subheader("Unsupervised Learning: Clustering Comparison")
        
        with st.spinner("Evaluating clustering performance..."):
            clustering_results = evaluate_clustering_performance(df_original, df_morphed)
        
        if 'error' not in clustering_results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Dataset Clustering:**")
                orig_clust = clustering_results['original']
                st.metric("Inertia", f"{orig_clust['inertia']:.2f}")
                if orig_clust['silhouette_score'] is not None:
                    st.metric("Silhouette Score", f"{orig_clust['silhouette_score']:.4f}")
                st.metric("Number of Clusters", orig_clust['n_clusters'])
            
            with col2:
                st.write("**Morphed Dataset Clustering:**")
                morph_clust = clustering_results['morphed']
                st.metric("Inertia", f"{morph_clust['inertia']:.2f}")
                if morph_clust['silhouette_score'] is not None:
                    st.metric("Silhouette Score", f"{morph_clust['silhouette_score']:.4f}")
                st.metric("Number of Clusters", morph_clust['n_clusters'])
            
            # Clustering differences
            st.subheader("Clustering Preservation")
            diffs = clustering_results['differences']
            st.metric("Inertia Difference", f"{diffs['inertia_diff']:.2f}")
            if 'silhouette_diff' in diffs:
                st.metric("Silhouette Score Difference", f"{diffs['silhouette_diff']:.4f}")
            
        else:
            st.error(f"Clustering evaluation failed: {clustering_results['error']}")
    
    # Overall preservation score
    if 'error' not in ml_results:
        st.header("Overall Preservation Score")
        
        preservation_score = calculate_preservation_score(stats_results, ml_results)
        
        if 'error' not in preservation_score:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                score = preservation_score['overall_score']
                interpretation = preservation_score['interpretation']
                
                # Create a progress bar for the score
                st.metric("Preservation Score", f"{score:.1f}/100")
                st.progress(score/100)
                st.success(f"**{interpretation}**")
                
                # Show component scores
                with st.expander("Detailed Score Breakdown"):
                    components = preservation_score['component_scores']
                    for component, score_val in components.items():
                        st.metric(component.replace('_', ' ').title(), f"{score_val:.1f}/100")
        
        else:
            st.error(preservation_score['error'])
    
    # Export options
    st.header("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Generate PDF Report", key="download_pdf_report"):
            pdf_buffer = generate_pdf_report(df_original, df_morphed, stats_results, ml_results, shape)
            st.download_button(
                label="📥 Download PDF",
                data=pdf_buffer,
                file_name=f"statmorph_report_{shape}.pdf",
                mime="application/pdf"
            )
    
    with col2:
        csv_data = create_csv_summary(stats_results, ml_results)
        st.download_button(
            label="Download CSV Summary",
            data=csv_data,
            file_name=f"statmorph_summary_{shape}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Download morphed data
        if st.button("Generate Final Dataset", key="download_morphed_final"):
            csv = df_morphed.to_csv(index=False)
            st.download_button(
                label="Download Morphed CSV",
                data=csv,
                file_name=f"morphed_data_{shape}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


def create_scatter_plot(df, title, color):
    """Create a scatter plot for the dataset."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['x'], df['y'], alpha=0.6, c=color, s=50, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    mean_x, mean_y = df['x'].mean(), df['y'].mean()
    ax.axvline(mean_x, color='red', linestyle='--', alpha=0.7, label=f'Mean X: {mean_x:.2f}')
    ax.axhline(mean_y, color='red', linestyle='--', alpha=0.7, label=f'Mean Y: {mean_y:.2f}')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def generate_pdf_report(df_original, df_morphed, stats_results, ml_results, shape):
    """Generate a PDF report of the analysis results."""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("StatMorph Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Summary
    summary_text = f"""
    <b>Analysis Summary</b><br/>
    Shape: {shape.title()}<br/>
    Original Dataset: {len(df_original)} points<br/>
    Morphed Dataset: {len(df_morphed)} points<br/>
    """
    summary = Paragraph(summary_text, styles['Normal'])
    story.append(summary)
    story.append(Spacer(1, 12))
    
    # Statistical Results Table
    stats_title = Paragraph("Statistical Comparison", styles['Heading2'])
    story.append(stats_title)
    
    # Convert DataFrame to table data
    table_data = [['Metric', 'Value']]
    for _, row in stats_results.iterrows():
        table_data.append([row['Metric'], row['Value']])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 12))
    
    # ML Results
    if 'error' not in ml_results:
        ml_title = Paragraph("Machine Learning Performance", styles['Heading2'])
        story.append(ml_title)
        
        ml_text = f"""
        Original Dataset Accuracy: {ml_results['original']['accuracy']:.4f}<br/>
        Morphed Dataset Accuracy: {ml_results['morphed']['accuracy']:.4f}<br/>
        Accuracy Difference: {ml_results['differences']['accuracy_diff']:.4f}<br/>
        """
        ml_paragraph = Paragraph(ml_text, styles['Normal'])
        story.append(ml_paragraph)
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def create_csv_summary(stats_results, ml_results):
    """Create a CSV summary of all results."""
    
    # Combine all results into a single summary
    summary_data = []
    
    # Add statistical results
    for _, row in stats_results.iterrows():
        summary_data.append({
            'Category': 'Statistical',
            'Metric': row['Metric'],
            'Value': row['Value']
        })
    
    # Add ML results if available
    if 'error' not in ml_results:
        ml_metrics = [
            ('Original Accuracy', ml_results['original']['accuracy']),
            ('Morphed Accuracy', ml_results['morphed']['accuracy']),
            ('Accuracy Difference', ml_results['differences']['accuracy_diff']),
            ('Original F1 Score', ml_results['original']['f1_score']),
            ('Morphed F1 Score', ml_results['morphed']['f1_score']),
            ('F1 Score Difference', ml_results['differences']['f1_diff'])
        ]
        
        for metric_name, value in ml_metrics:
            summary_data.append({
                'Category': 'Machine Learning',
                'Metric': metric_name,
                'Value': f"{value:.4f}" if isinstance(value, float) else str(value)
            })
    
    # Convert to DataFrame and then CSV
    summary_df = pd.DataFrame(summary_data)
    return summary_df.to_csv(index=False)


if __name__ == "__main__":
    main()