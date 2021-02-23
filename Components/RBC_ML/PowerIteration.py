import torch
import numpy as np

from Tests.Tools.saver import save_info_stuck


class PowerIteration:
    def __init__(self, device, dtype, max_error):
        self.device = device
        self.dtype = dtype
        self.max_error = max_error

    def eigenvalue(self, A, v):
        Av = torch.mm(A, v)
        return torch.dot(v.flatten(), Av.flatten())

    def power_iteration(self, A):
        max_loops = 1000
        n, d = A.shape
        v = (torch.ones(d) / np.sqrt(d)).to(device=self.device, dtype=self.dtype).view(d, 1)
        ev = self.eigenvalue(A, v)

        i = 0

        while True:
            i += 1
            Av = torch.mm(A, v)
            v_new = Av / torch.linalg.norm(Av.flatten())

            ev_new = self.eigenvalue(A, v_new)
            if torch.abs(ev - ev_new) < self.max_error:
                break
            if i > max_loops:
                raise Exception(f'stuck at power iteration more than {max_loops} loops')

            v = v_new
            ev = ev_new

        return ev_new, v_new.flatten()



