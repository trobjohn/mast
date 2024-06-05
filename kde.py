import numpy as np
import matplotlib.pyplot as plt

class KDensity:
    """ Transparent nonparametric analysis. """

    def __init__(self,x,kernel_type = 'Gaussian'):
        """ Initialize nonparametric estimator. """   
        self.tau = np.sqrt(2*np.pi)**(-.5)
        self.x = x
        self.n = len(self.x)
        self.kernel_type = kernel_type
        #
        self.h_rot = 1.06 * np.var(self.x) * len(self.x)**(-.2)
        self.h = self.h_rot
        self.h_min = max(.05,self.h_rot/100)
        self.data_dsq = self.dist_sq(x,x)
        #
        self.min_x = np.min(x)*.8
        self.max_x = np.max(x)*1.2
        self.grid = np.linspace(self.min_x,self.max_x,100)
        self.grid_dsq = self.dist_sq(self.grid,self.x)
        #
        self.h_lscv = None
        self.h_lcv = None
        self.f_hat = None

    def krnl(self,z):
        """ Compute kernel values for a vector z. """
        if self.kernel_type == 'Gaussian':
            y = self.tau * np.exp( - z/2 )
        return y

    def krnl_bar(self,z):
        """ Computes bar function for LSCV. """
        if self.kernel_type == 'Gaussian':
            kbar = np.exp(-z/4)/np.sqrt(4*np.pi)
        return kbar

    def dist_sq(self,x,z):
        """ Compute distance squared from x to z. """
        if type(x) == True:
            x = x.to_numpy()
        if type(z) == 'pandas.core.frame.DataFrame':
            z = z.to_numpy()
        dsq = ( np.tile(x,(len(z),1)).T - np.tile(z,(len(x),1)) )**2
        return dsq

    def compute_density(self,h=None):
        """ Computes density. """
        if h is None :
            h = self.h_rot
        m = self.krnl(self.grid_dsq/h)
        self.f_hat = np.sum( m, axis=1)/m.shape[1]

    def plot_density(self):
        """ Plots density. """
        print(self.grid)
        print(self.f_hat)
        plt.plot(self.grid,self.f_hat)
        plt.show()
        
    def lscv_fcn(self,h):
        """ Compute least squares cross validation function. """
        a_mat = self.krnl_bar(self.data_dsq/h**2)
        a = np.sum( np.sum(a_mat,axis=1) )/( (self.n**2) *h)
        b_mat = self.krnl(self.data_dsq/h**2)
        b = np.sum(np.sum(b_mat,axis=0))/( (self.n**2) *h) - np.sum(np.diag(b_mat))/(self.n*(self.n-1)*h)
        ls = a - 2*b
        return ls

    def lcv_fcn(self,h):
        """ Compute likelihood cross validation function. """
        loo_matrix = self.krnl(self.data_dsq/h**2)
        np.fill_diagonal(loo_matrix,0)
        f_hat_mi = np.sum( loo_matrix,axis=1)/( (self.n-1)*h)
        lik = np.sum( np.log(f_hat_mi))/self.n
        return lik

    def lscv(self, grid_points = 35, h_lb = None, h_ub = None):
        """ Perform least squares cross validation. """
        if h_lb is None:
            h_lb = self.h_min
        if h_ub is None:
            h_ub = self.h_rot*1.1
        h_grid = np.linspace(h_lb, h_ub, grid_points)
        sse = np.zeros(grid_points)
        #    
        for k in range(grid_points):
            sse[k] = self.lscv_fcn(h_grid[k])
        min_sse = min(sse)
        min_index = np.where( sse == min_sse )
        self.h_lscv = h_grid[min_index][0]
        return grid_points, sse

    def lcv(self, grid_points = 50, h_lb = None, h_ub = None):
        """ Perform likelihood cross validation. """
        if h_lb is None:
            h_lb = self.h_min
        if h_ub is None:
            h_ub = self.h_rot*1.5
        h_grid = np.linspace(h_lb, h_ub, grid_points)
        LL = np.zeros(grid_points)
        #
        for k in range(grid_points):
            LL[k] = self.lcv_fcn(h_grid[k])
        max_sse = max(LL)
        max_index = np.where( LL == max_sse )
        self.h_lscv = h_grid[max_index][0]
        return grid_points, LL
