import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy as np
import scipy as sp
from scipy.sparse import linalg


if __name__ == '__main__':

    M = nx.to_scipy_sparse_matrix(G, nodelist=list(G), weight=weight, dtype=float)
    eigenvalue, eigenvector = linalg.eigs(M.T, k=1, which="LR", maxiter=max_iter, tol=tol)
    largest = eigenvector.flatten().real
    norm = np.sign(largest.sum()) * sp.linalg.norm(largest)
    return dict(zip(G, largest / norm))

  # A = np.array([[1, 0], [0, -2]])
  # eigvals, eigvecs = la.eig(A)
  # eigvals = eigvals.real
  # lambda1 = eigvals[1]
  # v1 = eigvecs[:, 1].reshape(2, 1)
  # print(lambda1)