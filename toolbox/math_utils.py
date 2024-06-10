import numpy as np


###############################################################################

def normalize_by_std(vals):
    """Normalize so that mean is zero and standard deviation is 1"""
    vals = np.array(vals)
    mean = sum(vals)/len(vals)
    sd = np.std(vals)
    vals = vals - mean
    return vals/sd

