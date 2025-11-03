"""
Evaluator Module - Statistical and ML performance evaluation
Provides functions to compare original and morphed datasets for statistical integrity and ML performance.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


def evaluate_stats(df_orig, df_morph):
    """
    Evaluate statistical differences between original and morphed datasets.
    
    Parameters:
    -----------
    df_orig : pandas.DataFrame
        Original dataset with 'x' and 'y' columns
    df_morph : pandas.DataFrame
        Morphed dataset with 'x' and 'y' columns
        
    Returns:
    --------
    pandas.DataFrame
        Statistical comparison results including means, stds, and KS test p-values
    """
    
    # Calculate basic statistics
    orig_stats = {
        'x_mean': df_orig['x'].mean(),
        'x_std': df_orig['x'].std(),
        'y_mean': df_orig['y'].mean(),
        'y_std': df_orig['y'].std()
    }
    
    morph_stats = {
        'x_mean': df_morph['x'].mean(),
        'x_std': df_morph['x'].std(),
        'y_mean': df_morph['y'].mean(),
        'y_std': df_morph['y'].std()
    }
    
    # Calculate absolute differences
    mean_diff_x = abs(orig_stats['x_mean'] - morph_stats['x_mean'])
    mean_diff_y = abs(orig_stats['y_mean'] - morph_stats['y_mean'])
    std_diff_x = abs(orig_stats['x_std'] - morph_stats['x_std'])
    std_diff_y = abs(orig_stats['y_std'] - morph_stats['y_std'])
    
    # Perform Kolmogorov-Smirnov tests
    ks_stat_x, ks_p_x = stats.ks_2samp(df_orig['x'], df_morph['x'])
    ks_stat_y, ks_p_y = stats.ks_2samp(df_orig['y'], df_morph['y'])
    
    # Calculate correlation preservation
    orig_corr = df_orig['x'].corr(df_orig['y'])
    morph_corr = df_morph['x'].corr(df_morph['y'])
    corr_diff = abs(orig_corr - morph_corr)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Metric': [
            'X Mean (Original)',
            'X Mean (Morphed)',
            'X Mean Difference',
            'X Std (Original)',
            'X Std (Morphed)', 
            'X Std Difference',
            'Y Mean (Original)',
            'Y Mean (Morphed)',
            'Y Mean Difference',
            'Y Std (Original)',
            'Y Std (Morphed)',
            'Y Std Difference',
            'X-Y Correlation (Original)',
            'X-Y Correlation (Morphed)',
            'Correlation Difference',
            'KS Test X (p-value)',
            'KS Test Y (p-value)'
        ],
        'Value': [
            f"{orig_stats['x_mean']:.4f}",
            f"{morph_stats['x_mean']:.4f}",
            f"{mean_diff_x:.4f}",
            f"{orig_stats['x_std']:.4f}",
            f"{morph_stats['x_std']:.4f}",
            f"{std_diff_x:.4f}",
            f"{orig_stats['y_mean']:.4f}",
            f"{morph_stats['y_mean']:.4f}",
            f"{mean_diff_y:.4f}",
            f"{orig_stats['y_std']:.4f}",
            f"{morph_stats['y_std']:.4f}",
            f"{std_diff_y:.4f}",
            f"{orig_corr:.4f}",
            f"{morph_corr:.4f}",
            f"{corr_diff:.4f}",
            f"{ks_p_x:.4f}",
            f"{ks_p_y:.4f}"
        ]
    })
    
    return results


def evaluate_ml_comprehensive(df_orig, df_morph, test_size=0.3, random_state=42):
    """
    Comprehensive ML evaluation using multiple algorithms and metrics.
    
    Parameters:
    -----------
    df_orig : pandas.DataFrame
        Original dataset with 'x' and 'y' columns
    df_morph : pandas.DataFrame
        Morphed dataset with 'x' and 'y' columns
    test_size : float, default=0.3
        Proportion of dataset to use for testing
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Comprehensive ML performance comparison results
    """
    
    def prepare_data_and_evaluate_models(df, label_threshold=None):
        """Helper function to prepare data and evaluate multiple models."""
        
        # Generate synthetic binary labels based on threshold
        if label_threshold is None:
            label_threshold = df['x'].median() + df['y'].median()
        
        # Create binary classification target
        y = ((df['x'] + df['y']) > label_threshold).astype(int)
        X = df[['x', 'y']].values
        
        # Ensure we have both classes
        if len(np.unique(y)) < 2:
            # Adjust threshold to ensure balanced classes
            threshold_percentile = 50
            while len(np.unique(y)) < 2 and threshold_percentile > 10:
                label_threshold = np.percentile(df['x'] + df['y'], threshold_percentile)
                y = ((df['x'] + df['y']) > label_threshold).astype(int)
                threshold_percentile -= 10
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models to evaluate
        models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
            'SVM': SVC(random_state=random_state, probability=True),
            'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'Neural Network': MLPClassifier(random_state=random_state, max_iter=500)
        }
        
        # Neural Network hyperparameter tuning
        def tune_neural_network(X_train, y_train):
            """
            Perform hyperparameter tuning for Neural Network using GridSearchCV.
            
            Parameters:
            -----------
            X_train : array-like
                Training features
            y_train : array-like
                Training labels
                
            Returns:
            --------
            MLPClassifier
                Best tuned neural network model
            dict
                Best parameters found
            """
            # Define hyperparameter grid for neural network
            nn_param_grid = {
                'hidden_layer_sizes': [
                    (50,), (100,), (50, 30), (100, 50), (100, 50, 25)
                ],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'solver': ['adam', 'lbfgs']
            }
            
            # Create base neural network
            base_nn = MLPClassifier(random_state=random_state, max_iter=500)
            
            # Perform grid search with cross-validation
            grid_search = GridSearchCV(
                base_nn, 
                nn_param_grid, 
                cv=3,  # Use 3-fold CV for speed
                scoring='accuracy',
                n_jobs=-1,  # Use all available cores
                verbose=0
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            return grid_search.best_estimator_, grid_search.best_params_
        
        results = {}
        tuned_nn_params = None  # Store neural network tuning results
        
        for model_name, model in models.items():
            try:
                # Special handling for Neural Network with hyperparameter tuning
                if model_name == 'Neural Network':
                    print(f"Tuning hyperparameters for {model_name}...")
                    tuned_model, best_params = tune_neural_network(X_train_scaled, y_train)
                    tuned_nn_params = best_params
                    model = tuned_model
                    print(f"Best Neural Network parameters: {best_params}")
                
                # Train model (already trained if it's the tuned neural network)
                if model_name != 'Neural Network':
                    model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = None
                
                # Get prediction probabilities for AUC calculation
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted')
                
                # Calculate AUC if possible
                auc = None
                if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                    try:
                        auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        auc = None
                
                # Cross-validation score (use base model for CV to avoid re-tuning)
                if model_name == 'Neural Network':
                    # Use the tuned model for CV
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                model_result = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
                
                # Add hyperparameter info for neural network
                if model_name == 'Neural Network' and tuned_nn_params:
                    model_result['best_params'] = tuned_nn_params
                    model_result['architecture'] = tuned_nn_params.get('hidden_layer_sizes', 'N/A')
                    model_result['activation'] = tuned_nn_params.get('activation', 'N/A')
                    model_result['alpha'] = tuned_nn_params.get('alpha', 'N/A')
                
                results[model_name] = model_result
                
            except Exception as e:
                error_result = {
                    'error': str(e),
                    'accuracy': 0,
                    'f1_score': 0,
                    'precision': 0,
                    'recall': 0,
                    'auc': None,
                    'cv_mean': 0,
                    'cv_std': 0
                }
                
                if model_name == 'Neural Network':
                    error_result['best_params'] = None
                    error_result['architecture'] = 'Error'
                    error_result['activation'] = 'Error'
                    error_result['alpha'] = 'Error'
                
                results[model_name] = error_result
        
        return {
            'models': results,
            'n_samples': len(X),
            'n_positive': sum(y),
            'threshold': label_threshold
        }
    
    try:
        # Use the same threshold for both datasets for fair comparison
        threshold = df_orig['x'].median() + df_orig['y'].median()
        
        # Evaluate original dataset
        orig_results = prepare_data_and_evaluate_models(df_orig, threshold)
        
        # Evaluate morphed dataset
        morph_results = prepare_data_and_evaluate_models(df_morph, threshold)
        
        # Calculate differences for each model
        model_differences = {}
        for model_name in orig_results['models'].keys():
            if 'error' not in orig_results['models'][model_name] and 'error' not in morph_results['models'][model_name]:
                orig_model = orig_results['models'][model_name]
                morph_model = morph_results['models'][model_name]
                
                model_differences[model_name] = {
                    'accuracy_diff': abs(orig_model['accuracy'] - morph_model['accuracy']),
                    'f1_diff': abs(orig_model['f1_score'] - morph_model['f1_score']),
                    'precision_diff': abs(orig_model['precision'] - morph_model['precision']),
                    'recall_diff': abs(orig_model['recall'] - morph_model['recall']),
                    'cv_mean_diff': abs(orig_model['cv_mean'] - morph_model['cv_mean'])
                }
                
                if orig_model['auc'] is not None and morph_model['auc'] is not None:
                    model_differences[model_name]['auc_diff'] = abs(orig_model['auc'] - morph_model['auc'])
        
        results = {
            'original': orig_results,
            'morphed': morph_results,
            'model_differences': model_differences
        }
        
        return results
        
    except Exception as e:
        return {
            'error': f"Comprehensive ML evaluation failed: {str(e)}",
            'original': None,
            'morphed': None,
            'model_differences': None
        }


def evaluate_clustering_performance(df_orig, df_morph, n_clusters=3, random_state=42):
    """
    Evaluate clustering performance on original vs morphed datasets.
    
    Parameters:
    -----------
    df_orig : pandas.DataFrame
        Original dataset with 'x' and 'y' columns
    df_morph : pandas.DataFrame
        Morphed dataset with 'x' and 'y' columns
    n_clusters : int, default=3
        Number of clusters for K-means
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Clustering performance comparison results
    """
    
    def evaluate_clustering(df):
        """Helper function to evaluate clustering on a dataset."""
        
        X = df[['x', 'y']].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        try:
            silhouette = silhouette_score(X_scaled, cluster_labels)
        except:
            silhouette = None
        
        # Calculate cluster centers and sizes
        unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        cluster_centers = kmeans.cluster_centers_
        
        return {
            'inertia': inertia,
            'silhouette_score': silhouette,
            'n_clusters': len(unique_labels),
            'cluster_sizes': cluster_sizes.tolist(),
            'cluster_centers': cluster_centers.tolist()
        }
    
    try:
        # Evaluate both datasets
        orig_clustering = evaluate_clustering(df_orig)
        morph_clustering = evaluate_clustering(df_morph)
        
        # Calculate differences
        differences = {
            'inertia_diff': abs(orig_clustering['inertia'] - morph_clustering['inertia']),
        }
        
        if orig_clustering['silhouette_score'] is not None and morph_clustering['silhouette_score'] is not None:
            differences['silhouette_diff'] = abs(orig_clustering['silhouette_score'] - morph_clustering['silhouette_score'])
        
        return {
            'original': orig_clustering,
            'morphed': morph_clustering,
            'differences': differences
        }
        
    except Exception as e:
        return {
            'error': f"Clustering evaluation failed: {str(e)}"
        }


def evaluate_ml(df_orig, df_morph, test_size=0.3, random_state=42):
    """
    Evaluate ML performance on original vs morphed datasets (simplified version for compatibility).
    
    Parameters:
    -----------
    df_orig : pandas.DataFrame
        Original dataset with 'x' and 'y' columns
    df_morph : pandas.DataFrame
        Morphed dataset with 'x' and 'y' columns
    test_size : float, default=0.3
        Proportion of dataset to use for testing
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict
        ML performance comparison results
    """
    
    def prepare_data_and_train(df, label_threshold=None):
        """Helper function to prepare data and train model."""
        
        # Generate synthetic binary labels based on threshold
        if label_threshold is None:
            label_threshold = df['x'].median() + df['y'].median()
        
        # Create binary classification target
        y = ((df['x'] + df['y']) > label_threshold).astype(int)
        X = df[['x', 'y']].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=random_state, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'n_samples': len(X),
            'n_positive': sum(y),
            'threshold': label_threshold
        }
    
    try:
        # Use the same threshold for both datasets for fair comparison
        threshold = df_orig['x'].median() + df_orig['y'].median()
        
        # Evaluate original dataset
        orig_results = prepare_data_and_train(df_orig, threshold)
        
        # Evaluate morphed dataset
        morph_results = prepare_data_and_train(df_morph, threshold)
        
        # Calculate differences
        results = {
            'original': orig_results,
            'morphed': morph_results,
            'differences': {
                'accuracy_diff': abs(orig_results['accuracy'] - morph_results['accuracy']),
                'f1_diff': abs(orig_results['f1_score'] - morph_results['f1_score']),
                'precision_diff': abs(orig_results['precision'] - morph_results['precision']),
                'recall_diff': abs(orig_results['recall'] - morph_results['recall'])
            }
        }
        
        return results
        
    except Exception as e:
        return {
            'error': f"ML evaluation failed: {str(e)}",
            'original': None,
            'morphed': None,
            'differences': None
        }


def create_comprehensive_ml_table(ml_results):
    """
    Create a formatted table for comprehensive ML performance comparison.
    
    Parameters:
    -----------
    ml_results : dict
        Results from evaluate_ml_comprehensive function
        
    Returns:
    --------
    pandas.DataFrame
        Formatted comprehensive comparison table
    """
    
    if 'error' in ml_results:
        return pd.DataFrame({
            'Error': [ml_results['error']]
        })
    
    orig_models = ml_results['original']['models']
    morph_models = ml_results['morphed']['models']
    differences = ml_results['model_differences']
    
    # Create comprehensive comparison
    rows = []
    
    for model_name in orig_models.keys():
        if 'error' not in orig_models[model_name]:
            orig = orig_models[model_name]
            morph = morph_models[model_name]
            diff = differences.get(model_name, {})
            
            rows.append({
                'Model': model_name,
                'Metric': 'Accuracy',
                'Original': f"{orig['accuracy']:.4f}",
                'Morphed': f"{morph['accuracy']:.4f}",
                'Difference': f"{diff.get('accuracy_diff', 'N/A'):.4f}" if 'accuracy_diff' in diff else 'N/A'
            })
            
            rows.append({
                'Model': model_name,
                'Metric': 'F1 Score',
                'Original': f"{orig['f1_score']:.4f}",
                'Morphed': f"{morph['f1_score']:.4f}",
                'Difference': f"{diff.get('f1_diff', 'N/A'):.4f}" if 'f1_diff' in diff else 'N/A'
            })
            
            rows.append({
                'Model': model_name,
                'Metric': 'CV Mean',
                'Original': f"{orig['cv_mean']:.4f}",
                'Morphed': f"{morph['cv_mean']:.4f}",
                'Difference': f"{diff.get('cv_mean_diff', 'N/A'):.4f}" if 'cv_mean_diff' in diff else 'N/A'
            })
            
            if orig['auc'] is not None:
                rows.append({
                    'Model': model_name,
                    'Metric': 'AUC',
                    'Original': f"{orig['auc']:.4f}",
                    'Morphed': f"{morph['auc']:.4f}",
                    'Difference': f"{diff.get('auc_diff', 'N/A'):.4f}" if 'auc_diff' in diff else 'N/A'
                })
            
            # Add Neural Network specific information
            if model_name == 'Neural Network' and 'best_params' in orig:
                if orig['best_params'] is not None:
                    rows.append({
                        'Model': model_name,
                        'Metric': 'Architecture (Original)',
                        'Original': str(orig.get('architecture', 'N/A')),
                        'Morphed': str(morph.get('architecture', 'N/A')),
                        'Difference': 'Config Info'
                    })
                    
                    rows.append({
                        'Model': model_name,
                        'Metric': 'Activation Function',
                        'Original': str(orig.get('activation', 'N/A')),
                        'Morphed': str(morph.get('activation', 'N/A')),
                        'Difference': 'Config Info'
                    })
                    
                    rows.append({
                        'Model': model_name,
                        'Metric': 'Regularization (Alpha)',
                        'Original': str(orig.get('alpha', 'N/A')),
                        'Morphed': str(morph.get('alpha', 'N/A')),
                        'Difference': 'Config Info'
                    })
    
    return pd.DataFrame(rows)


def create_neural_network_details_table(ml_results):
    """
    Create a detailed table showing Neural Network hyperparameter tuning results.
    
    Parameters:
    -----------
    ml_results : dict
        Results from evaluate_ml_comprehensive function
        
    Returns:
    --------
    pandas.DataFrame
        Neural Network configuration details
    """
    
    if 'error' in ml_results:
        return pd.DataFrame({
            'Error': [ml_results['error']]
        })
    
    orig_models = ml_results['original']['models']
    morph_models = ml_results['morphed']['models']
    
    if 'Neural Network' not in orig_models:
        return pd.DataFrame({
            'Info': ['Neural Network not found in results']
        })
    
    orig_nn = orig_models['Neural Network']
    morph_nn = morph_models['Neural Network']
    
    # Create detailed neural network comparison
    rows = []
    
    if 'best_params' in orig_nn and orig_nn['best_params'] is not None:
        orig_params = orig_nn['best_params']
        morph_params = morph_nn.get('best_params', {})
        
        for param_name, orig_value in orig_params.items():
            morph_value = morph_params.get(param_name, 'N/A')
            
            rows.append({
                'Hyperparameter': param_name.replace('_', ' ').title(),
                'Original Dataset': str(orig_value),
                'Morphed Dataset': str(morph_value),
                'Status': 'Same' if orig_value == morph_value else 'Different'
            })
        
        # Add performance summary
        rows.append({
            'Hyperparameter': '--- Performance Summary ---',
            'Original Dataset': '---',
            'Morphed Dataset': '---',
            'Status': '---'
        })
        
        rows.append({
            'Hyperparameter': 'Final Accuracy',
            'Original Dataset': f"{orig_nn['accuracy']:.4f}",
            'Morphed Dataset': f"{morph_nn['accuracy']:.4f}",
            'Status': f"Δ {abs(orig_nn['accuracy'] - morph_nn['accuracy']):.4f}"
        })
        
        rows.append({
            'Hyperparameter': 'Cross-Validation Mean',
            'Original Dataset': f"{orig_nn['cv_mean']:.4f}",
            'Morphed Dataset': f"{morph_nn['cv_mean']:.4f}",
            'Status': f"Δ {abs(orig_nn['cv_mean'] - morph_nn['cv_mean']):.4f}"
        })
    
    return pd.DataFrame(rows)
    """
    Evaluate ML performance on original vs morphed datasets.
    
    Parameters:
    -----------
    df_orig : pandas.DataFrame
        Original dataset with 'x' and 'y' columns
    df_morph : pandas.DataFrame
        Morphed dataset with 'x' and 'y' columns
    test_size : float, default=0.3
        Proportion of dataset to use for testing
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict
        ML performance comparison results
    """
    
    def prepare_data_and_train(df, label_threshold=None):
        """Helper function to prepare data and train model."""
        
        # Generate synthetic binary labels based on threshold
        if label_threshold is None:
            label_threshold = df['x'].median() + df['y'].median()
        
        # Create binary classification target
        y = ((df['x'] + df['y']) > label_threshold).astype(int)
        X = df[['x', 'y']].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=random_state, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'n_samples': len(X),
            'n_positive': sum(y),
            'threshold': label_threshold
        }
    
    try:
        # Use the same threshold for both datasets for fair comparison
        threshold = df_orig['x'].median() + df_orig['y'].median()
        
        # Evaluate original dataset
        orig_results = prepare_data_and_train(df_orig, threshold)
        
        # Evaluate morphed dataset
        morph_results = prepare_data_and_train(df_morph, threshold)
        
        # Calculate differences
        results = {
            'original': orig_results,
            'morphed': morph_results,
            'differences': {
                'accuracy_diff': abs(orig_results['accuracy'] - morph_results['accuracy']),
                'f1_diff': abs(orig_results['f1_score'] - morph_results['f1_score']),
                'precision_diff': abs(orig_results['precision'] - morph_results['precision']),
                'recall_diff': abs(orig_results['recall'] - morph_results['recall'])
            }
        }
        
        return results
        
    except Exception as e:
        return {
            'error': f"ML evaluation failed: {str(e)}",
            'original': None,
            'morphed': None,
            'differences': None
        }


def create_ml_comparison_table(ml_results):
    """
    Create a formatted table for ML performance comparison.
    
    Parameters:
    -----------
    ml_results : dict
        Results from evaluate_ml function
        
    Returns:
    --------
    pandas.DataFrame
        Formatted comparison table
    """
    
    if 'error' in ml_results:
        return pd.DataFrame({
            'Error': [ml_results['error']]
        })
    
    orig = ml_results['original']
    morph = ml_results['morphed']
    diff = ml_results['differences']
    
    comparison_df = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'F1 Score',
            'Precision',
            'Recall',
            'Training Samples',
            'Positive Class Ratio'
        ],
        'Original Dataset': [
            f"{orig['accuracy']:.4f}",
            f"{orig['f1_score']:.4f}",
            f"{orig['precision']:.4f}",
            f"{orig['recall']:.4f}",
            f"{orig['n_samples']}",
            f"{orig['n_positive']/orig['n_samples']:.3f}"
        ],
        'Morphed Dataset': [
            f"{morph['accuracy']:.4f}",
            f"{morph['f1_score']:.4f}",
            f"{morph['precision']:.4f}",
            f"{morph['recall']:.4f}",
            f"{morph['n_samples']}",
            f"{morph['n_positive']/morph['n_samples']:.3f}"
        ],
        'Absolute Difference': [
            f"{diff['accuracy_diff']:.4f}",
            f"{diff['f1_diff']:.4f}",
            f"{diff['precision_diff']:.4f}",
            f"{diff['recall_diff']:.4f}",
            "0",  # Same number of samples
            f"{abs(orig['n_positive']/orig['n_samples'] - morph['n_positive']/morph['n_samples']):.3f}"
        ]
    })
    
    return comparison_df


def calculate_preservation_score(stats_df, ml_results):
    """
    Calculate an overall preservation score based on statistical and ML metrics.
    
    Parameters:
    -----------
    stats_df : pandas.DataFrame
        Statistical comparison results
    ml_results : dict
        ML evaluation results
        
    Returns:
    --------
    dict
        Preservation scores and interpretation
    """
    
    if 'error' in ml_results:
        return {'error': 'Cannot calculate preservation score due to ML evaluation error'}
    
    try:
        # Extract key difference values
        mean_diffs = [
            float(stats_df[stats_df['Metric'] == 'X Mean Difference']['Value'].iloc[0]),
            float(stats_df[stats_df['Metric'] == 'Y Mean Difference']['Value'].iloc[0])
        ]
        
        std_diffs = [
            float(stats_df[stats_df['Metric'] == 'X Std Difference']['Value'].iloc[0]),
            float(stats_df[stats_df['Metric'] == 'Y Std Difference']['Value'].iloc[0])
        ]
        
        corr_diff = float(stats_df[stats_df['Metric'] == 'Correlation Difference']['Value'].iloc[0])
        
        # KS test p-values (higher is better)
        ks_p_values = [
            float(stats_df[stats_df['Metric'] == 'KS Test X (p-value)']['Value'].iloc[0]),
            float(stats_df[stats_df['Metric'] == 'KS Test Y (p-value)']['Value'].iloc[0])
        ]
        
        # ML performance difference
        ml_diff = ml_results['differences']['accuracy_diff']
        
        # Calculate scores (0-100 scale)
        mean_score = max(0, 100 - np.mean(mean_diffs) * 1000)  # Scale factor for visibility
        std_score = max(0, 100 - np.mean(std_diffs) * 1000)
        corr_score = max(0, 100 - corr_diff * 100)
        ks_score = min(100, np.mean(ks_p_values) * 100)  # Higher p-value is better
        ml_score = max(0, 100 - ml_diff * 100)
        
        # Overall score (weighted average)
        overall_score = (mean_score * 0.2 + std_score * 0.2 + corr_score * 0.2 + 
                        ks_score * 0.2 + ml_score * 0.2)
        
        # Interpretation
        if overall_score >= 90:
            interpretation = "Excellent preservation"
        elif overall_score >= 75:
            interpretation = "Good preservation"
        elif overall_score >= 60:
            interpretation = "Moderate preservation"
        else:
            interpretation = "Poor preservation"
        
        return {
            'overall_score': overall_score,
            'interpretation': interpretation,
            'component_scores': {
                'mean_preservation': mean_score,
                'std_preservation': std_score,
                'correlation_preservation': corr_score,
                'distribution_similarity': ks_score,
                'ml_performance_preservation': ml_score
            }
        }
        
    except Exception as e:
        return {'error': f'Error calculating preservation score: {str(e)}'}


def create_neural_network_architecture_summary(ml_results):
    """
    Create a summary of neural network architectures and their performance.
    
    Parameters:
    -----------
    ml_results : dict
        Results from evaluate_ml_comprehensive function
        
    Returns:
    --------
    dict
        Neural network architecture summary
    """
    
    if 'error' in ml_results:
        return {'error': ml_results['error']}
    
    orig_models = ml_results['original']['models']
    morph_models = ml_results['morphed']['models']
    
    if 'Neural Network' not in orig_models:
        return {'error': 'Neural Network not found in results'}
    
    orig_nn = orig_models['Neural Network']
    morph_nn = morph_models['Neural Network']
    
    summary = {
        'original_config': {},
        'morphed_config': {},
        'performance_comparison': {},
        'architecture_info': {}
    }
    
    # Extract configuration details
    if 'best_params' in orig_nn and orig_nn['best_params'] is not None:
        orig_params = orig_nn['best_params']
        morph_params = morph_nn.get('best_params', {})
        
        summary['original_config'] = orig_params
        summary['morphed_config'] = morph_params
        
        # Architecture analysis
        orig_layers = orig_params.get('hidden_layer_sizes', (100,))
        morph_layers = morph_params.get('hidden_layer_sizes', (100,))
        
        summary['architecture_info'] = {
            'original_layers': orig_layers,
            'morphed_layers': morph_layers,
            'original_total_neurons': sum(orig_layers) if isinstance(orig_layers, tuple) else orig_layers,
            'morphed_total_neurons': sum(morph_layers) if isinstance(morph_layers, tuple) else morph_layers,
            'original_depth': len(orig_layers) if isinstance(orig_layers, tuple) else 1,
            'morphed_depth': len(morph_layers) if isinstance(morph_layers, tuple) else 1
        }
        
        # Performance comparison
        summary['performance_comparison'] = {
            'accuracy_orig': orig_nn['accuracy'],
            'accuracy_morph': morph_nn['accuracy'],
            'accuracy_diff': abs(orig_nn['accuracy'] - morph_nn['accuracy']),
            'cv_mean_orig': orig_nn['cv_mean'],
            'cv_mean_morph': morph_nn['cv_mean'],
            'cv_mean_diff': abs(orig_nn['cv_mean'] - morph_nn['cv_mean']),
            'f1_orig': orig_nn['f1_score'],
            'f1_morph': morph_nn['f1_score'],
            'f1_diff': abs(orig_nn['f1_score'] - morph_nn['f1_score'])
        }
    
    return summary