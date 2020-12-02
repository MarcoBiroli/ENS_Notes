import numpy as np
import scipy.sparse as sparse

class Operators:
    def __init__(self, dim = 0, N = [0], normalization = None, boundary_conditions = 'periodic', stencil_type = 5):
        '''
        Container class for all useful differential operators. dim = 1, 2, 3 is the dimension
        of the space we are working in. N is the number of grid_points in each direction, 
        normalization = [dx, dy, dz] is the normalization. boundary_conditions = 'periodic', 
        'hard' are the two possible boundary conditions. stencil_type = 5, 9 defines the stencil
        used for the Laplacian.
        '''
        self.dim = dim
        assert len(N) == self.dim, 'Dimensions of grid_points do not match dimension of the space.'
        self.N = N
        self.M = np.prod(self.N)
        if normalization is None:
            self.normalization = [1 for _ in range(self.dim)]
        else:
            assert len(normalization) == self.dim, 'Dimensions of normalization do not match dimension of the space.'
            self.normalization = normalization
        self.boundary_conditions = boundary_conditions
        self.directional_grads = [self._computetwopointgrad(i) for i in range(self.dim)]
        self.grad = self._computegrad()
        self.laplacian = self._computelaplacian(stencil_type)

    def reindexing(self, point):
        '''
        Computes the re-indexation from a 3D point array to a 1D array.
        '''
        assert len(point) == self.dim, 'Point dimension invalid'
        if self.dim == 1:
            return point[0]
        elif self.dim == 2:
            # (x, y) -> (x + y * self.X_length)
            return point[0] + point[1] * self.N[0]
        elif self.dim == 3:
            # (x, y, z) -> (x + self.X_length(y + self.Y_length * z))
            return point[0] + self.N[0] * (point[1] + self.N[1] * point[2])


    def _computetwopointgrad(self, dir = 0):
        '''
        Computes the one directional derivative along the given direction.
        '''
        assert dir < self.dim, 'The given direction must be smaller than the dimensions of the problem.'
        alpha =  2 * self.normalization[dir] ** (-2)
        displacement = self.reindexing([1 if i == dir else 0 for i in range(self.dim)])
        if self.boundary_conditions == 'periodic':
            return sparse.diags([-alpha, alpha, -alpha, alpha], [-displacement, displacement, \
                self.M - displacement, displacement - self.M], shape = (self.M, self.M), format = 'csr')
        elif self.boundary_conditions == 'hard':
            return sparse.diags([-alpha, alpha], [-displacement, displacement], shape = (self.M, self.M), format = 'csr')
    
    def _computegrad(self):
        def apply_grad(u):
            return np.concatenate([(self.directional_grads[i] @ u).reshape(self.M, 1) for i in range(self.dim)], \
                axis = 1)
        return lambda u : apply_grad(u)
    
    def _computelaplacian(self, stencil_type):
        assert stencil_type in [5, 9], 'Invalid stencil type.'
        if stencil_type == 5:
            return self._computelaplacian5point()
        elif stencil_type == 9:
            return self._computelaplacian9point()
            

    def _computelaplacian5point(self):
        alphas = [di**(-2) for di in self.normalization]
        beta = - 2 * np.sum(alphas)
        if self.boundary_conditions == 'periodic':
            diagonals = [beta]
            for alpha in alphas:
                diagonals += [alpha]*4
            displacements = [0]
            for i in range(self.dim):
                dspl = self.reindexing([1 if j == i else 0 for j in range(self.dim)])
                displacements += [dspl, -dspl, self.M - dspl, dspl - self.M]
            return sparse.diags(diagonals, displacements, shape = (self.M, self.M), format='csr')
        if self.boundary_conditions == 'hard':
            diagonals = [beta]
            for alpha in alphas:
                diagonals += [alpha]*2
            displacements = [0]
            for i in range(self.dim):
                dspl = self.reindexing([1 if j == i else 0 for j in range(self.dim)])
                displacements += [dspl, -dspl]
            return sparse.diags(diagonals, displacements, shape = (self.M, self.M), format='csr')

    def _computelaplacian9point(self):
        raise ValueError('TBD.')
