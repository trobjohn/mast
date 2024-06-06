import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import norm


class Probit:
    """ Probit regression analysis kit. """
    def __init__(self,x,y,include_intercept=True):
        """ Initialize probit regression. """
        # Filter missings.
        check_x = np.isnan(x)
        check_y = np.isnan(y)
        keep_row = ( (np.sum(check_x,axis=1)+check_y) == 0)
        n = len(y)
        NAs = n - np.sum(keep_row)
        if NAs>0 :
            self.keep_row = keep_row
            self.x = x.loc[keep_row,:].to_numpy()
            self.y = y[keep_row].to_numpy()
            print(f'Due to missing values, {NAs} observations out of {n} were dropped.')
        else:
            self.keep_row = None
            self.x = x.to_numpy()
            self.y = y.to_numpy()
        #   
        self.depvar = y.name
        self.vars = x.columns
        #
        self.n = len(self.y)
        self.ybar = np.mean(self.y)
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
        self.mem = None
        self.ame = None
        #
        self.se = None
        #
        self.beta = None
        self.vcv = None
        self.se = None
        self.pval = None
        self.t_stat = None
        #
        self.latent = None
        self.y_hat = None
        #
        self.output = None

    def F(self,z):
        """ Standard normal distribution function. """
        r = norm.cdf(z)
        return r
    
    def f(self,z):
        """ Standard normal density function. """
        r = norm.pdf(z)
        return r
    
    def mills(self,z):
        """ Mills ratio for standard normal. """
        eps = 1e-10
        den = eps+self.F(z)
        num = self.f(z)
        #r = np.zeros(len(z))
        #select = np.isinf(den)
        #r[ select ] = np.inf 
        #r[ select == False ] = num[select == False]/ den[select == False]
        r = num/den
        return r

    def eval(self,beta,hessian=False):
        delta = 1e-50
        latent = self.x@beta
        F_1 = self.F(latent)
        F_0 = self.F(-latent)
        mills_1 = self.mills(latent)
        mills_0 = self.mills(-latent)
        # Log likelihood:
        LL = np.sum( self.y*np.log(F_1+delta) + (1-self.y)*np.log(F_0+delta) )/self.n
        # Gradient:
        gr = self.x.T @ ( self.y*mills_1 - (1-self.y)*mills_0 )/self.n
        # Hessian:
        if hessian is True:
            eta_0 = - mills_0 * latent - mills_0**2
            eta_1 = - mills_1 * (-latent) - mills_1**2 
            jac = self.x.T * ( self.y*mills_1 - (1-self.y)*mills_0 )
            D = np.diag( self.y*eta_1 + (1-self.y)*eta_0 ) 
            H =  self.x.T @ D @ self.x
        else:
            H = None
            jac = None
        #
        return LL, gr, H, jac

    def solve_gd(self,eps=1e-11,max_itr=500):
        """ Estimate beta by gradient descent. """
        err = 10
        eta = .01
        beta = (1/self.k)*np.ones(self.k)
        itr = 0
        #
        while err>eps and itr < max_itr:
            LL, gr, H, jac = self.eval(beta)
            if itr>0:
                num = np.abs( np.inner(beta - beta_m1, gr - gr_m1))
                den = np.sum( (gr-gr_m1)**2 )
                eta = num/den
            beta_p1 = beta + eta*gr
            err = np.sqrt(np.sum(gr**2))
            itr = itr+1
            beta_m1 = np.copy(beta)
            beta = np.copy(beta_p1)
            gr_m1 = np.copy(gr)
        print(f"Final error for iteration {itr} is {err}")
        self.beta = beta
        self.latent = self.x @ self.beta
        self.y_hat = self.F(self.latent)

    def compute_se(self, se_type = 'standard'):
        """ Compute standard errors."""
        self.se_type = se_type
        LL, gr, H, jac = self.eval(self.beta,hessian=True)
        if se_type == 'standard':
            self.vcv = np.linalg.inv(-H)
            self.se = np.sqrt(np.diag(self.vcv))
        elif se_type == 'robust':
            print('Robust standard errors')
            meat = jac @ jac.T #jac @ np.ones((self.n,self.n)) @ jac.T # suspect
            bread = np.linalg.inv(-H)
            self.vcv = bread @ meat @ bread
            self.se = np.sqrt(np.diag(self.vcv))

    def summary(self,digits=4):
        """ Print regression output. """
        self.t_stat = self.beta/self.se
        self.pval = 2*(t_dist.cdf(-abs(self.t_stat), self.n-self.k))
        #
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
                            'Significance': stars,
                            'MEM':self.mem,
                            'AME':self.ame
                            })
        output = np.around(output,decimals=digits)
        return output
    
    def mfx(self,type='AME'):
        """ Compute marginal effects. """
        #
        latent = self.x@self.beta
        f = self.f(latent)
        self.ame = np.mean(f) * self.beta
        #
        latent = np.mean(self.x,axis=0)@self.beta
        f = self.f(latent)
        self.mem = f*self.beta
        
    def run(self,se_type = 'standard'):
        """ Run regression and print results. """
        self.solve_gd()
        self.compute_se(se_type)
        self.mfx()
        self.output = self.summary()

        with open('log.txt','w') as f:
            print(self.output,file=f)

        print('\n', self.output, '\n')



