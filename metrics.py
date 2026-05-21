import numpy as np
import scipy.spatial as sp


#Drift metrics

def CKA(r1, r2, ):
    """
    r1: first representation to compare. Should be a 2D array of shape (m_inputs, n_features)
    r2: second representation to compare. Should be a 2D array of shape (m_inputs, n_features)
    """
    
    #Acquire gram matrices
    K = r1 @ r1.T
    L = r2 @ r2.T
    
    #Center K an L
    K_centered = K - K.mean(axis=0) - K.mean(axis=1)[:, np.newaxis] + K.mean()
    L_centered = L - L.mean(axis=0) - L.mean(axis=1)[:, np.newaxis] + L.mean()
    
    #Compute CKA
    cka = hsic(K_centered, L_centered) / np.sqrt(hsic(K_centered, K_centered) * hsic(L_centered, L_centered))
    return cka

def hsic(X, Y):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) for the given centered gram matrices.
    X and Y are centered gram matrices.
    """
    
    m = X.shape[0]
    return np.sum(X * Y) / (m - 1)**2

def cosine_similarity(r1, r2):
    """
    Compute cosine similarity for matrix representations.
    r1 and r2 are 2D arrays of shape (m_inputs, n_features).
    """
    
    #r1 and r2 magnitudes
    r1_magnitude = np.linalg.norm(r1, axis=1)
    r2_magnitude = np.linalg.norm(r2, axis=1)
    
    #Dot product
    dot_product = np.sum(r1 * r2, axis=1)
    
    #Cosine similarity
    cosine_sim = dot_product / (r1_magnitude * r2_magnitude + 1e-12)  # Add small value to avoid division by zero
    return cosine_sim.mean()  # Return average cosine similarity across all inputs
    
def participation_ratio(r):
    """
    Compute the participation ratio of a representation r.
    r is a 2D array of shape (m_inputs, n_features).
    """
    #eigenvalues of the covariance matrix
    cov_matrix = np.cov(r, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    
    #PR
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return pr
