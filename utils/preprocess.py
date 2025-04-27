import numpy as np

def extract_features(password):
    return np.array([
        len(password),                         # Length
        sum(c.isupper() for c in password),    # Uppercase count
        sum(c.isdigit() for c in password),    # Digits count
        sum(not c.isalnum() for c in password) # Special chars count
    ]).reshape(1, -1)