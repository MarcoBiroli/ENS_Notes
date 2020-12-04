import numpy as np

class Wavefunction:
    '''
    Wavefunction class with some helpful methods.
    '''
    def __init__(self, u = None, dim = None, N = None):
        '''
        Parameters are:
            - u : initial value of the wavefunction. 
            - dim : dimension of the space (1, 2, or 3)
            - N : number of points in each space dimensions: [N_x, N_y, N_z] in 3D for example
        Note that u must have shape (N_1, N_2, ..., N_dim) or (N_1 * N_2 * ... * N_dim, 1). 
        '''
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
        '''
        Copy function that returns a new wavefunction with the same values and parameters. 
        '''
        return Wavefunction(self.u.copy(), self.dim, self.N)

    def _computenormsq_pointwise(self):
        '''
        Computes the norm squared of the wavefunction at every point in space
        '''
        if self.u is None:
            return None
        return np.real(np.conjugate(self.u) * self.u)

    def _computenorm(self):
        '''
        Computes the norm of the wavefunction
        '''
        if self.u is None:
            return None
        return np.sqrt(np.sum(np.real(np.conjugate(self.u) * self.u)))

    def computeCurrent(self, Ops, fact = 1/40):
        '''
        Computes the probability current for this wavefunction.
        '''
        assert (self.dim == Ops.dim and self.N == Ops.N), 'Incompatible wavefunction and operators.'
        gradu = Ops.grad(self.u)
        graduconj = Ops.grad(np.conjugate(self.u))
        out = (np.conjugate(self.u) * gradu - self.u * graduconj)/1j
        normout = np.max(np.sqrt(out[:, 0]**2 + out[:, 1]**2))
        #return fact*np.real(out)
        return fact*np.real(out/normout)

    def update(self, new_u):
        '''
        Update the values for this wavefunction.
        '''
        self.u = new_u # Change the values
        self.norm = self._computenorm() # Compute the norm
        self.u /= self.norm # Normalize the wavefunction
        if self.u is not None:
            self.u = self.u.reshape(len(self.u), 1)
        self.normsq_pointwise = self._computenormsq_pointwise() # Compute the new norms
        self.norm = self._computenorm()

    def laplacian(self, Ops):
        '''
        Returns a wavefunction containing the laplacian of the current wavefunction. Ops must be an
        instance of the Operator class with parameters matching the current wavefunction.
        '''
        assert (self.dim == Ops.dim and self.N == Ops.N), 'Incompatible wavefunction and operators.'
        return Wavefunction(Ops.laplacian.dot(self.u), self.dim, self.N)

