from . import polystate as ps
import numpy as np


N=8
indices = np.arange(N) - (N - 1) / 2
parity = (np.indices((N, N)).sum(axis=0)) % 2
mask = parity.astype(bool)
X, Y = np.meshgrid(indices, indices)
chess_rts = ( (X[mask]) + 1j * (Y[mask]) ).flatten()  

def random_chess(rc):
    indices = np.random.choice(len(chess_rts), size=rc, replace=False)
    sampled_values = chess_rts[indices]
    t1 = np.random.rand(rc)
    t2 = np.random.rand(rc)
    random_complex = t1 + 1j * t2
    result = sampled_values + random_complex
    return result

def random_chess1(rc,t1,t2):
    indices = np.random.choice(len(chess_rts), size=rc, replace=False)
    sampled_values = chess_rts[indices]
    random_complex = t1 + 1j * t2
    result = sampled_values + random_complex
    return result
