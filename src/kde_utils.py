import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def compute_kde_from_features(features, bandwidths=[0.01, 0.05, 0.1, 0.5, 1]):
    """
    Compute the KDE from the features with cross-validation for bandwidth selection.
    
    Parameters:
    - features: List of feature vectors
    - bandwidths: List of bandwidths to try for KDE
    
    Returns:
    - Fitted KDE model
    - Selected bandwidth
    """
    # Convert features to numpy array if not already
    features = np.array(features)
    
    # Use grid search to find the best bandwidth
    grid = GridSearchCV(
        KernelDensity(kernel='gaussian'),
        {'bandwidth': bandwidths},
        cv=min(5, len(features))  # Use 5-fold CV or less if fewer samples
    )
    grid.fit(features)
    
    # Get the best bandwidth
    best_bandwidth = grid.best_params_['bandwidth']
    
    # Fit KDE with best bandwidth
    kde = KernelDensity(bandwidth=best_bandwidth, kernel='gaussian')
    kde.fit(features)
    
    return kde, best_bandwidth

def get_likelihood_ratio(features, val_kde, test_kde):
    """
    Compute likelihood ratio between test KDE and validation KDE.
    
    Parameters:
    - features: Feature vectors to compute ratio for
    - val_kde: KDE model fitted on validation data
    - test_kde: KDE model fitted on test data
    
    Returns:
    - Array of likelihood ratios
    """
    val_density = val_kde.score_samples(features)
    test_density = test_kde.score_samples(features)
    return np.exp(test_density) / np.exp(val_density) 