"""
Graph-complexity analysis for the simulation frames.
Boltzmann entropy is now computed directly in simulation.py.
"""

from collections import Counter

import cv2
import networkx as nx
import numpy as np
from skimage.color import rgb2lab

COMPLEXITY_SIZE = (10, 10)   # 100 pixels — consistent with the rest of the project
THRESHOLD = 0.5


def resize_frame(frame, size=COMPLEXITY_SIZE):
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def frame_to_lab_features(frame):
    return rgb2lab(frame.astype(np.float32) / 255.0).reshape(-1, 3)


def compute_global_stats(all_features):
    stacked = np.vstack(all_features)
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def compute_similarity_matrix(features, global_mean, global_std):
    normed   = (features - global_mean) / global_std
    print("features")
    print(features)
    print("normed")
    print(normed)
    sq_norms = (normed ** 2).sum(axis=1)
    print("sq_norms")
    print(sq_norms)
    dist_sq  = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (normed @ normed.T)
    np.clip(dist_sq, 0, None, out=dist_sq)
    sim = 1.0 - np.sqrt(dist_sq) / np.sqrt(normed.shape[1])
    np.clip(sim, 0, 1, out=sim)
    return sim


def build_graph(sim_matrix, threshold=THRESHOLD):
    n = sim_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    rows, cols = np.where(sim_matrix > threshold)
    mask = rows < cols
    rows, cols = rows[mask], cols[mask]
    G.add_weighted_edges_from(
        zip(rows.tolist(), cols.tolist(), sim_matrix[rows, cols].tolist())
    )
    return G


def distinct_degree_types(G):
    return len(set(d for _, d in G.degree()))
