import numpy as np
from Quant import wavefunc as wf
from differentials import differentials as df
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import progressbar

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
        #self.psi = wf.Wavefunction(np.ones(self.M)/np.sqrt(self.M), self.dim, self.N, self.normalization)
        if self.dim == 1:
            self.psi = wf.Wavefunction((np.exp(-self.grids**2/2)/np.sqrt(np.pi) ).ravel(), self.dim, self.N, self.normalization)
        elif self.dim == 2:
            self.psi = wf.Wavefunction((np.exp(-(self.grids[0]**2+self.grids[1]**2)/(2.0)) /np.sqrt(np.pi) ).ravel() , self.dim, self.N, self.normalization)
        #self.phi = self.psi.copy()
        #self.phi.update(np.absolute(self.phi.u))
        # Define the operators that we will use.
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
        #print(load.shape)
        self.psi.update(u)
        

    def create_step_matrix(self, ImTime):
        '''
        This function computes the matrices A and Aprime such that the Crank Nicolson methods can be written as:
        A @ psi_{n+1} = Aprime @ psi_{n}. The argument ImTime is a bool indicating whether we are doing the imaginary
        time evolution (ImTime = True) or the real time evolution (ImTime = False). 
        '''
        # According to the type of time evolution we can get an extra "i" factor.
        if ImTime:
            fact = 1
        else:
            fact = 1j
        # Diagonal portion of A
        beta = (1/self.dt + \
                fact/2*(self.U + self.interC * (np.absolute(self.psi.u)**2) ) )
        # Diagonal portion of Aprime
        betaprev = (1/self.dt - \
                fact/2*(self.U + self.interC * (np.absolute(self.psi.u)**2) ) )
        # Building A and Aprime without their laplatian part.
        A = sparse.diags([beta.ravel()], [0], shape=(self.M, self.M), format = 'csr')
        Aprev = sparse.diags([betaprev.ravel()], [0], shape=(self.M, self.M), format = 'csr')
        return A + fact*self.laplatianportion, Aprev - fact*self.laplatianportion # Return the full expression of A and Aprime.

    def compute_mu(self):
        '''
        # This function computes the chemical potential of the current wavefunction. 
        '''
        laplatianu = self.psi.laplacian(self.Ops).u
        ekin = np.sum((-self.hbar**2/(2*self.m) * laplatianu * np.conjugate(self.psi.u))) * np.prod(self.normalization)
        epot = np.sum(self.U * self.psi.normsq_pointwise ) * np.prod(self.normalization)
        eint = self.interC * np.sum(self.psi.normsq_pointwise**2) * np.prod(self.normalization)
        #print(ekin - ekin_K, epot - epot_K, eint - eint_K)
        #out = np.sum((-self.hbar**2/(2*self.m) * laplatianu * np.conjugate(self.psi.u) + self.U * np.absolute(self.psi.u)**2 \
        #    + self.interC/2 * np.absolute(self.psi.u)**4)*np.prod(self.normalization))
        mu = ekin + epot + eint
        energy = ekin + epot + 0.5*eint
        return mu, energy

    def update_wavefunction(self, A, Aprev):
        '''
        Updates the wavefunciton by performing one step of the Crank Nicolson algorithm given the step matrices,
        A and Aprime computed in the "create_step_matrix" method.
        '''
        # Since the matrices are sparse and diagonal in nature we use scipy's spsolve function to solve the problem.
        new_u = splinalg.spsolve(A, Aprev.dot(self.psi.u).flatten())
        # Update the wavefunction with the newly computed value.
        self.psi.update(new_u)

    def im_time_evolution(self):
        '''
        Performs the intial imaginary time evolution to reach the ground state. Returns the evolution
        of the chemical potential.
        '''
        # initialize a progress bar for display purposes.
        bar = progressbar.ProgressBar(maxval=self.im_iters)
        mutab = []
        print('\nStarting imaginary time convergence ...')
        for cnt in range(self.im_iters):
            # For each imaginary time iteration perform one step of the Crank Nicolson algorithm
            if cnt%10 == 0:
                bar.update(cnt)
            #self.phi.update(2*self.psi.normsq_pointwise - self.phi.u)
            A, Aprev = self.create_step_matrix(True) # ImTime = True
            self.update_wavefunction(A, Aprev)
            mutab.append(self.compute_mu()) # Compute the chemical potential and append it to the list
        return mutab # Return the evolution of the Chemical potential which should converge to a stable value.

    def real_time_evolution(self):
        '''
        Performs the real time evolution of the problem and returns the list of encountered wavefunctions
        and currents.
        '''
        utab = []
        jtab = []
        # intialize a progress bar for display purposes
        bar = progressbar.ProgressBar(maxval=len(self.t))
        bar_cnt = 0
        print('\nStarting real time evolution ...')
        for cur_t in self.t:
            # For each time step perform one step of the Crank Nicolson algorithm
            bar.update(bar_cnt)
            bar_cnt += 1
            self.U = self.potential(self.grids, cur_t).reshape(self.M, 1) #Compute the new potential and flatten it
            #self.phi.update(2*self.psi.normsq_pointwise - self.phi.u)
            A, Aprev = self.create_step_matrix(False) # ImTime = False
            self.update_wavefunction(A, Aprev)
            utab.append(self.psi.copy()) # Append encountered wavefunction to the list
            j = self.psi.computeCurrent(self.Ops) #Compute the current and append it to the list
            jtab.append(j.copy())
        return utab, jtab # Return the list of states and currents.

    def full_solve(self, save = (False, '')):
        '''
        This function calls all the necessary functions one after the other and returns the evolution
        of the chemical potential during the imaginary time convergence, and the wavefunctions and currents
        encountered in the real time evolution.
        '''
        mutab = self.im_time_evolution()
        if save[0]:
            np.save(save[1], self.psi.u)
        utab, jtab = self.real_time_evolution()
        return mutab, utab, jtab 

    def winding_number(self, point, rad, psi = None):
        if psi == None:
            j = self.psi.computeCurrent(self.Ops, fact = 1)
        else:
            j = psi.computeCurrent(self.Ops, fact = 1)
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
                #print(pos[0], pos[1])
                #print(integral, j[pos[1], pos[0]], self.grids[0][pos[1], pos[0]], self.grids[1][pos[1], pos[0]])
                nextpos = [(pos[0] + 1)%self.N[0], pos[1]]
                integral += (j[pos[1], pos[0], 0] + j[nextpos[1], nextpos[0], 0])*self.normalization[0]/2
                pos = nextpos.copy()
                path.append(pos)
            #input()
            for _ in range(2*radidx[1]):
                #print(integral, j[pos[1], pos[0]], self.grids[0][pos[1], pos[0]], self.grids[1][pos[1], pos[0]])
                nextpos = [pos[0], (pos[1] + 1)%self.N[1]]
                integral += (j[pos[1], pos[0], 1] + j[nextpos[1], nextpos[0], 1])*self.normalization[1]/2
                pos = nextpos.copy()
                path.append(pos)
            #input()
            for _ in range(2*radidx[0]):
                #print(integral, j[pos[1], pos[0]], self.grids[0][pos[1], pos[0]], self.grids[1][pos[1], pos[0]])
                nextpos = [(pos[0] - 1)%self.N[0], pos[1]]
                integral -= (j[pos[1], pos[0], 0] + j[nextpos[1], nextpos[0], 0])*self.normalization[0]/2
                pos = nextpos.copy()
                path.append(pos)
            #input()
            for _ in range(2*radidx[1]):
                #print(integral, j[pos[1], pos[0]], self.grids[0][pos[1], pos[0]], self.grids[1][pos[1], pos[0]])
                nextpos = [pos[0], (pos[1] - 1)%self.N[1]]
                integral -= (j[pos[1], pos[0], 1] + j[nextpos[1], nextpos[0], 1])*self.normalization[1]/2
                pos = nextpos.copy()
                path.append(pos)
            #input()
            return integral, path
        elif self.dim == 3:
            raise NotImplementedError
        
