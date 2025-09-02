import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    i = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, i])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self

    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


def compute_normals(points, k_neighbors=10):
    """
    Calculate the normal vectors for each point in a point cloud using PCA, supports batch processing.

    Parameters:
    - points: A torch tensor of shape [batch_size, n_points, 3] representing the batch of point clouds.
    - k_neighbors: Number of nearest neighbors to consider for estimating the normal at each point.

    Returns:
    - normals: A torch tensor of shape [batch_size, n_points, 3] containing the normal vectors.
    """
    batch_size, n_points, _ = points.shape
    normals = []
    
    # Process each point cloud in the batch
    for b in range(batch_size):
        batch_normals = []
        for i in range(n_points):
            # Calculate distances from the current point to all other points
            distances = torch.sum((points[b] - points[b, i]) ** 2, axis=1)
            # Find indices of the k nearest neighbors (excluding the point itself)
            _, k_neighbor_idxs = torch.topk(distances, k_neighbors+1, largest=False)
            k_neighbor_idxs = k_neighbor_idxs[k_neighbor_idxs != i][0:k_neighbors]

            # Apply PCA to find the normals using the neighbors
            pca = PCA(n_components=3)
            neighbor_points = points[b, k_neighbor_idxs]
            pca.fit(neighbor_points)
            normal = pca.components_[-1]

            # Ensure consistent direction of normals (pointing outwards)
            if torch.dot(normal, points[b, i]) > 0:
                normal = -normal

            batch_normals.append(normal.unsqueeze(0))

        normals.append(torch.cat(batch_normals, dim=0))

    return torch.stack(normals, dim=0)  # Convert back to tensor