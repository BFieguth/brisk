# algorithms.py
import brisk
                
ALGORITHM_CONFIG = brisk.AlgorithmCollection(
    *brisk.REGRESSION_ALGORITHMS,
    *brisk.CLASSIFICATION_ALGORITHMS    
)
