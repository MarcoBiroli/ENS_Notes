import numpy as np

class Wavefunction:
    def __init__(self, u = None, dim = None, N = None):
        self.u = u
        self.dim = dim
        self.N = N
        if u is not None:
            self.u = self.u.reshape(len(self.u), 1)
        else:
            self.N = 0
        self.normsq_pointwise = self._computenormsq_pointwise()
        self.norm = self._computenorm()
    
    def copy(self):
        return Wavefunction(self.u.copy(), self.dim, self.N)

    def _computenormsq_pointwise(self):
        if self.u is None:
            return None
        return np.real(np.conjugate(self.u) * self.u)

    def _computenorm(self):
        if self.u is None:
            return None
        return np.sqrt(np.sum(np.real(np.conjugate(self.u) * self.u)))

    def computeCurrent(self, Ops):
        assert (self.dim == Ops.dim and self.N == Ops.N), 'Incompatible wavefunction and operators.'
        gradu = Ops.grad(self.u)
        graduconj = Ops.grad(np.conjugate(self.u))
        out = (np.conjugate(self.u) * gradu - self.u * graduconj)/1j
        normout = np.max(np.sqrt(out[:, 0]**2 + out[:, 1]**2))
        return np.real(out/normout)

    def update(self, new_u):
        self.u = new_u
        self.norm = self._computenorm()
        self.u /= self.norm
        if self.u is not None:
            self.u = self.u.reshape(len(self.u), 1)
        self.normsq_pointwise = self._computenormsq_pointwise()
        self.norm = self._computenorm()

    def laplacian(self, Ops):
        assert (self.dim == Ops.dim and self.N == Ops.N), 'Incompatible wavefunction and operators.'
        return Wavefunction(Ops.laplacian.dot(self.u), self.dim, self.N)

