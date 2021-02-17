import torch
import numpy as np

DTYPE = torch.float
DEVICE = torch.device("cuda:0")


def eigenvalue(A, v):
    Av = torch.mm(A, v)
    return torch.dot(v.flatten(), Av.flatten())


def power_iteration(A, max_error):
    n, d = A.shape
    v = (torch.ones(d) / np.sqrt(d)).to(device=DEVICE, dtype=DTYPE).view(d, 1)
    ev = eigenvalue(A, v)

    i = 0

    while True:
        i += 1
        Av = torch.mm(A, v)
        v_new = Av / torch.linalg.norm(Av.flatten())

        ev_new = eigenvalue(A, v_new)
        if torch.abs(ev - ev_new) < max_error:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new.flatten()
