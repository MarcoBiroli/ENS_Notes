import numpy as np
from Quant import wavefunc as wf
from differentials import differentials as df
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import progressbar

class GPE_Solver:
    def __init__(self, dim, bounds, tmin, tmax, dt, N, im_iters, potential, interC, boundary_conditions):
        self.dim, self.bounds, self.tmin, self.tmax = dim, bounds, tmin, tmax
        self.dt, self.N, self.im_iters, self.potential = dt, N, im_iters, potential
        self.boundary_conditions = boundary_conditions
        self.M = np.prod(self.N)
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
        self.psi = wf.Wavefunction(np.ones(self.M)/np.sqrt(self.M), self.dim, self.N)
        self.Ops = df.Operators(self.dim, self.N, self.normalization, self.boundary_conditions, stencil_type = 5)
        self.U = potential(self.grids, 0).reshape(self.M, 1)
        self.interC = interC

    def create_step_matrix(self, ImTime):
        if ImTime:
            fact = 1
        else:
            fact = 1j
        laplatianportion = - fact* self.hbar**2 / (4 * self.m) * self.Ops.laplacian
        beta = (1/self.dt + \
                fact/2*(self.U + self.interC * (self.psi.u * np.conjugate(self.psi.u)) ) )
        betaprev = (1/self.dt - \
                fact/2*(self.U + self.interC * (self.psi.u * np.conjugate(self.psi.u)) ) )
        A = sparse.diags([beta.flatten()], [0], shape=(self.M, self.M), format = 'csr')
        Aprev = sparse.diags([betaprev.flatten()], [0], shape=(self.M, self.M), format = 'csr')
        return A + laplatianportion, Aprev - laplatianportion

    def compute_mu(self):
        laplatianu = self.psi.laplacian(self.Ops)
        return np.sum(1/2 * laplatianu.normsq_pointwise + self.U * self.psi.normsq_pointwise \
            + self.interC * self.psi.normsq_pointwise**2)

    def update_wavefunction(self, A, Aprev):
        new_u = splinalg.spsolve(A, Aprev.dot(self.psi.u).flatten())
        self.psi.update(new_u)

    def im_time_evolution(self):
        bar = progressbar.ProgressBar(max_value=self.im_iters)
        mutab = []
        print('\nStarting imaginary time convergence ...')
        for cnt in range(self.im_iters):
            bar.update(cnt)
            A, Aprev = self.create_step_matrix(True)
            self.update_wavefunction(A, Aprev)
            mutab.append(self.compute_mu())
        return mutab

    def real_time_evolution(self):
        utab = []
        jtab = []
        bar = progressbar.ProgressBar(max_value=len(self.t))
        bar_cnt = 0
        print('\nStarting real time evolution ...')
        for cur_t in self.t:
            bar.update(bar_cnt)
            bar_cnt += 1
            self.U = self.potential(self.grids, cur_t).reshape(self.M, 1)
            A, Aprev = self.create_step_matrix(False)
            self.update_wavefunction(A, Aprev)
            utab.append(self.psi.copy())
            j = self.psi.computeCurrent(self.Ops)
            jtab.append(j.copy())
        return utab, jtab

    def full_solve(self):
        mutab = self.im_time_evolution()
        utab, jtab = self.real_time_evolution()
        return mutab, utab, jtab