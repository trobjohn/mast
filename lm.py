import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

class LM:
    """ Linear regression analysis kit. """
    def __init__(self,x,y,include_intercept=True):
        """ Initialize linear regression. """
        # Filter missings.
        check_x = np.isnan(x.to_numpy())
        check_y = np.isnan(y.to_numpy())
        keep_row = ( (np.sum(check_x,axis=1)+check_y) == 0)
        n = len(y)
        NAs = n - np.sum(keep_row)
        if NAs>0 :
            self.keep_row = keep_row
            self.x = x.loc[keep_row,:].to_numpy()
            self.y = y[keep_row].to_numpy()
            print(f'Due to missing values, {NAs} observations out of {n} were dropped. \n')
        else:
            self.keep_row = None
            self.x = x.to_numpy()
            self.y = y.to_numpy()
        #   
        self.depvar = y.name
        self.vars = x.columns
        #
        self.n = len(self.y)
        self.include_intercept = include_intercept
        self.se_type = 'standard'
        #
        self.n = (self.x).shape[0]
        if self.include_intercept is True:
            #self.x.loc[:,'Intercept'] = np.ones(self.n)
            self.x = np.append(self.x, np.ones( (self.n,1) ),axis=1)
            self.vars = np.append(self.vars,'Intercept')
        self.k = self.x.shape[1]
        
        #
        self.xPx = (self.x.T @ self.x )
        #
        self.residuals = None
        self.sse = None
        self.mse = None
        self.rmse = None
        self.ser = None
        self.rsq = None
        self.se = None
        self.ybar = np.mean(self.y)
        #
        self.beta = None
        self.vcv = None
        self.se = None
        self.pval = None
        self.n_clusters = None
        self.cluster_varname = None
        self.t_stat = None
        #
        self.output = None

    def z_norm(self,w):
        """ z-Score normalize a vector w. """ 
        r = w - np.mean(w)
        s = np.sqrt( np.inner(r,r)/(len(r)-1) )
        z = r/s
        return z 

    def solve_normal(self):
        """ Estimate beta by solving normal equations. """ 
        # Run regression
        xPy = self.x.T @ self.y # Compute X'y
        self.beta = np.linalg.solve( self.xPx, xPy) # Solve normal equations
        #
        y_hat = self.x @ self.beta # Compute predictions
        residuals = self.y-y_hat # Compute residuals
        #
        self.sse =  np.inner(residuals,residuals) # Compute SSE
        self.mse = self.sse/self.n
        self.rmse = np.sqrt(self.mse)
        self.ser = np.sum(self.sse)/(self.n-self.k) # Compute standard error of regression
        tss = np.sum( ( self.y-np.mean(self.y) )**2 )
        self.rsq = 1 - self.sse/tss 
        self.residuals = self.y-y_hat # Compute residuals        

    def solve_gd(self,eps=10e-5,max_itr=500):
        """ Estimate beta by gradient descent. """
        err = 10
        eta = .01
        beta = np.ones(self.k)
        itr = 0
        while err>eps and itr < max_itr:
            gr = -self.x.T @ ( self.y - self.x @ beta)/self.n
            if itr>0:
                num = np.abs( np.inner(beta - beta_m1, gr - gr_m1))
                den = np.sum( (gr-gr_m1)**2 )
                eta = num/den
            beta_p1 = beta - eta*gr
            err = max( np.sum((beta_p1 - beta)**2),
                      np.sqrt(np.sum(gr**2)))
            itr = itr+1
            beta_m1 = np.copy(beta)
            beta = np.copy(beta_p1)
            gr_m1 = np.copy(gr)
        print(f"Final error for iteration {itr} is {err}")
        self.beta = beta
                #
        y_hat = self.x @ self.beta # Compute predictions
        self.residuals = self.y-y_hat # Compute residuals

    def compute_se(self, se_type='', cluster_var = None):
        """ Compute standard errors. Supports standard and robust. """
        self.sse =  np.inner(self.residuals,self.residuals) # Compute SSE
        self.mse = self.sse/self.n
        self.rmse = np.sqrt(self.mse)
        self.ser = np.sum(self.sse)/(self.n-self.k) # Compute standard error of regression
        tss = np.sum( ( self.y-np.mean(self.y) )**2 )
        self.rsq = 1 - self.sse/tss 
        #
        if se_type == '':
            self.se_type = ''
            self.vcv = self.ser * np.linalg.inv(self.xPx)
            self.se = np.sqrt(np.diag(self.vcv))
        elif se_type == 'robust':
            self.se_type = 'Robust'
            e_sq = self.residuals**2
            bread = np.linalg.inv(self.xPx)
            meat = self.x.T @ np.diag(e_sq) @ self.x
            self.vcv = bread @ meat @ bread * (self.n/(self.n-self.k))
            self.se = np.sqrt(np.diag(self.vcv))
        elif se_type == 'cluster-robust':
            self.se_type = 'Cluster-Robust'
            if not self.keep_row is None:
                cluster_var = cluster_var.loc[self.keep_row]
            cluster_groups = cluster_var.unique().tolist()
            G = len(cluster_groups)
            self.n_clusters = G
            self.cluster_varname = cluster_var.name
            #
            meat = np.empty((self.k,self.k))
            for g in range(G): # Cycle through G clusters
                select = (cluster_var == cluster_groups[g]) #####
                x_g = self.x[select,:]
                u_g = self.residuals[select]
                meat = meat + np.outer( (x_g.T @ u_g), (x_g.T @ u_g) )
            bread = np.linalg.inv(self.xPx)
            self.vcv = bread @ meat @ bread * ( (G/(G-1)) * (self.n-1)/(self.n-self.k))
            self.se = np.sqrt(np.diag(self.vcv))

    def summary(self):
        """ Print regression output. """
        self.t_stat = self.beta/self.se
        self.pval = 2*(t_dist.cdf(-abs(self.t_stat), self.n-self.k))
        nstars = np.zeros(len(self.pval))
        nstars[ self.pval <= .01 ] = 3
        nstars[ (self.pval > .01) * (self.pval <= .05) ] = 2
        nstars[ (self.pval > .05) * (self.pval <= .10) ] = 1
        stars = ['*'*int(i) for i in nstars]
        #
        output = pd.DataFrame({'Variable':self.vars,
                            'Coefficient':self.beta,
                            'Std.Err':self.se,
                            't-Statistic':self.t_stat,
                            'p-Value': self.pval,
                            'Significance': stars
                            })
        return output
    
    def run(self,
            se_type = '',
            use_gr = False,
            cluster_var = None):
        """ Run regression and print results. """
        if use_gr is True:
            self.solve_gd()
        else:
            self.solve_normal()
        #
        if se_type == '':
            self.compute_se(se_type)
        elif se_type == 'robust':
            print('Robust standard errors')
            self.compute_se(se_type)
        elif se_type == 'cluster-robust':
            print('Cluster-robust standard errors')
            self.compute_se(se_type,cluster_var)
        #
        self.output = self.summary()
        with open('log.txt','a') as f:
            print(self.output,file=f)
        #
        print('\n',self.summary(),'\n')