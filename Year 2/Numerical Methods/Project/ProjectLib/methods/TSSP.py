import numpy as np
from Quant import wavefunc as wf
from differentials import differentials as df
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import progressbar
from numpy import fft

class GPE_Solver:
    def __init__(self, dim, bounds, tmin, tmax, dt, N, im_iters, potential, interC, boundary_conditions):
        '''
        GPE-Crank Nicolson solver.
        Inputs are:
            dim - dimension of the problem. Can be 1, 2 or 3
            bounds - [[xmin, xmax]] in 1D, [[xmin, xmax], [ymin, ymax]] in 2D and so on ...
            tmin - initial time
            tmax - final time
            dt - time step
            N - number of points per space dimension: [N_x, N_y, N_z] in 3D for example. 
            Therefore total amount of grid points are N_x*N_y*N_z
            im_iters - number of imaginary iterations performed initially
            potential - The potential used for the Gross Pitaevskii equation. This has to be a function
            of the following shape: lambda bounds, t: potential(grids, t). Where "grids" is a list of 
            meshgrids ([xx, yy]) compatible with the "bounds" given before and "t" is the current time.
            interC - coefficient of the non-linear term of the GPE.
            boundary_conditions - selected boundary conditions: "periodic" or "hard". 
        '''
        self.dim, self.bounds, self.tmin, self.tmax = dim, bounds, tmin, tmax
        self.dt, self.N, self.im_iters, self.potential = dt, N, im_iters, potential
        self.boundary_conditions = boundary_conditions
        self.M = np.prod(self.N) # Total size of the flattened vector
        self.hbar = 1
        self.m = 1
        self.spaces = [np.linspace(self.bounds[i][0], self.bounds[i][1], self.N[i]) for i in range(self.dim)]
        if self.dim == 1:
            self.grids = self.spaces[0]
        elif self.dim == 2:
            self.grids = np.meshgrid(self.spaces[0], self.spaces[1])
        elif self.dim == 3:
            self.grids = np.meshgrid(self.spaces[0], self.spaces[1], self.spaces[2])
        self.t = np.arange(self.tmin, self.tmax, self.dt)
        self.normalization = [self.spaces[i][1] - self.spaces[i][0] for i in range(self.dim)]
        # Define the wavefunction that we will use.
        self.psi = wf.Wavefunction((np.exp(-(self.grids[0]**2+self.grids[1]**2)/(2.0)) /np.sqrt(np.pi) ).ravel() , self.dim, self.N, self.normalization)
        self.Ops = df.Operators(self.dim, self.N, self.normalization, self.boundary_conditions, stencil_type = 5)
        self.U = potential(self.grids, self.tmin).reshape(self.M, 1) #Compute initial potential and flatten it.
        self.interC = interC
        self.laplatianportion = - self.hbar**2 / (4 * self.m) * self.Ops.laplacian

    def load_init_state(self, filename, ImFilename = None):
        load = np.loadtxt(filename, delimiter=',')
        u = load
        if ImFilename != None:
            load2 = np.loadtxt(ImFilename, delimiter = ',')
            u += 1j*load2
        self.psi.update(u)
        
    def compute_mu(self):
        '''
        # This function computes the chemical potential of the current wavefunction. 
        '''
        laplatianu = self.psi.laplacian(self.Ops).u
        ekin = np.sum((-self.hbar**2/(2*self.m) * laplatianu * np.conjugate(self.psi.u))) * np.prod(self.normalization)
        epot = np.sum(self.U * self.psi.normsq_pointwise ) * np.prod(self.normalization)
        eint = self.interC * np.sum(self.psi.normsq_pointwise**2) * np.prod(self.normalization)
        mu = ekin + epot + eint
        energy = ekin + epot + 0.5*eint
        return mu, energy

    def winding_number(self, point, rad):
        j = self.psi.computeCurrent(self.Ops, fact = 1)
        idx = []
        radidx = []
        for i in range(self.dim):
            idx.append(  int((point[i] - self.bounds[i][0])/self.normalization[i])  )
            radidx.append(   int(rad[i]/self.normalization[i])   )
        integral = 0
        if self.dim == 2:
            j = j.reshape(self.N[0], self.N[1], 2)
            pos = [(idx[0] - radidx[0])%self.N[0], (idx[1] - radidx[1])%self.N[1]]
            path = [pos]
            for _ in range(2*radidx[0]):
                print(integral)
                nextpos = [(pos[0] + 1)%self.N[0], pos[1]]
                integral += (j[pos[0], pos[1], 0] + j[nextpos[0], nextpos[1], 0])*self.normalization[0]
                pos = nextpos.copy()
                path.append(pos)
            for _ in range(2*radidx[1]):
                print(integral)
                nextpos = [pos[0], (pos[1] + 1)%self.N[1]]
                integral += (j[pos[0], pos[1], 1] + j[nextpos[0], nextpos[1], 1])*self.normalization[1]
                pos = nextpos.copy()
                path.append(pos)
            for _ in range(2*radidx[0]):
                print(integral)
                nextpos = [(pos[0] - 1)%self.N[0], pos[1]]
                integral -= (j[pos[0], pos[1], 0] + j[nextpos[0], nextpos[1], 0])*self.normalization[0]
                pos = nextpos.copy()
                path.append(pos)
            for _ in range(2*radidx[1]):
                print(integral)
                nextpos = [pos[0], (pos[1] - 1)%self.N[1]]
                integral -= (j[pos[0], pos[1], 1] + j[nextpos[0], nextpos[1], 1])*self.normalization[1]
                pos = nextpos.copy()
                path.append(pos)
            return integral, path
        elif self.dim == 3:
            raise NotImplementedError