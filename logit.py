import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

class Logit:
    """ Logistic regression analysis kit. """
    def __init__(self,x,y,
                 x_cat = None,
                 include_intercept=True):
        """ Initialize logistic regression. """
        ## Create x_cat dummies and add to x
        if x_cat is None:
            self.vars_cat = None
        else:
            dummies, labels = self.make_dummies(x_cat)
            x = pd.concat([x,pd.DataFrame(data=dummies,columns=labels)],axis=1)
        #
        ## Filter missings:
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
        ## Add intercept:
        self.ybar = np.mean(self.y)
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
        self.mem = None
        self.ame = None
        #
        self.se = None
        self.se_ame = None
        self.se_mem = None
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


    def F_logit(self,z):
        r = 1/(1+np.exp(-z))
        return r
    

    def eval(self,beta,z = None, hessian=False):
        if z is None:
            z = self.x
        delta = 0 # 1e-50
        latent = z@beta
        F_1 = self.F_logit(latent)
        F_0 = self.F_logit(-latent)
        # Log likelihood:
        LL = np.sum( self.y*np.log(F_1+delta) + (1-self.y)*np.log(F_0+delta) )/self.n
        # Gradient:
        gr = z.T @ ( self.y - F_1 )/self.n
        # Hessian and Jacobian for SE's:
        if hessian is True:
            jac = self.x.T * ( self.y - F_1 ) 
            D = np.diag(F_1*F_0)
            H = -(z.T) @ D @ z
        else:
            H = None
            jac = None
        #
        return LL, gr, H, jac


    def solve_gd(self,eps=1e-5,max_itr=1000):
        """ Estimate beta by gradient descent. """
        ## Iteration params
        err = 10
        eta = .01
        beta = (0)*np.ones(self.k)
        itr = 0
        #
        ## Transform data
        mu_x = np.mean(self.x,axis=0)
        sigma_x = np.sqrt( np.var(self.x,axis=0))
        intercept = np.where(sigma_x == 0 )
        if len(intercept)>1 :
            print('Warning: More than one regression variable is constant.')
        m_x = mu_x.copy()
        s_x = sigma_x.copy()
        m_x[intercept] = 0
        s_x[intercept] = 1
        z = (self.x-m_x)/s_x
        #
        ## Gradient descent
        while err>eps and itr < max_itr:
            LL, gr, H, jac = self.eval(beta,z=z)
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
        #
        ## Transform coefficients
        beta_star = beta.copy()
        selector = np.delete( range(len(mu_x)), intercept)
        beta_star[selector] = beta_star[selector]/sigma_x[selector]
        beta_star[intercept] = beta_star[intercept] - np.sum( beta[selector]*mu_x[selector]/sigma_x[selector])
        self.beta = beta_star
        #
        ## Set final values
        self.latent = self.x @ self.beta
        self.y_hat = self.F_logit(self.latent)

    def make_dummies(self,x_cat):
        """ Make dummy variables from categorical variables. """
        dummies = []
        labels = []
        for k in range(x_cat.shape[1]):
            lab_k,lev_k = np.unique(x_cat.iloc[:,k],return_inverse=True)
            dum_k = np.eye(len(lab_k))[lev_k]
            dum_k = dum_k[:,1:]
            dummies.append(dum_k)
            labels.append( [ x_cat.columns[k]+'_'+str(arr) for arr in lab_k ][1:] )
        dummies = np.concatenate(dummies,axis=1)
        labels = np.concatenate(labels)
        return dummies, labels
    
    def compute_se(self, se_type = ''):
        """ Compute standard errors."""
        self.se_type = se_type
        LL, gr, H, jac = self.eval(self.beta,hessian=True)
        if se_type == '':
            self.vcv = np.linalg.inv(-H)
            self.se = np.sqrt(np.diag(self.vcv))
        elif se_type == 'robust':
            print('Robust standard errors')
            meat = jac @ jac.T #jac @ np.ones((self.n,self.n)) @ jac.T # suspect
            bread = np.linalg.inv(-H)
            self.vcv = bread @ meat @ bread
            self.se = np.sqrt(np.diag(self.vcv))


    def summary(self,digits=4,display='coef'):
        """ Print regression output. """
        if display == 'coef':
            print('Displaying regression coefficients: \n')
            self.t_stat = self.beta/self.se
            self.pval = 2*(t_dist.cdf(-abs(self.t_stat), self.n-self.k))
            nstars = np.zeros(len(self.pval))
            nstars[ self.pval <= .01 ] = 3
            nstars[ (self.pval > .01) * (self.pval <= .05) ] = 2
            nstars[ (self.pval > .05) * (self.pval <= .10) ] = 1
            stars = ['*'*int(i) for i in nstars]
            #
            output = pd.DataFrame({'Var.':self.vars,
                                'Coef.':self.beta,
                                'Std.Err.':self.se,
                                't-Stat.':self.t_stat,
                                'p-Val.': self.pval,
                                'Stars': stars
                                })
        elif display == 'ame':
            print('Displaying average marginal effect (AME): \n')
            self.t_stat = self.ame/self.se_ame
            self.pval = 2*(t_dist.cdf(-abs(self.t_stat), self.n-self.k))
            nstars = np.zeros(len(self.pval))
            nstars[ self.pval <= .01 ] = 3
            nstars[ (self.pval > .01) * (self.pval <= .05) ] = 2
            nstars[ (self.pval > .05) * (self.pval <= .10) ] = 1
            stars = ['*'*int(i) for i in nstars]
            #
            output = pd.DataFrame({'Var.':self.vars,
                                'AME':self.ame,
                                'Std.Err.':self.se_ame,
                                't-Stat.':self.t_stat,
                                'p-Val.': self.pval,
                                'Stars': stars
                                })
        elif display == 'mem':
            print('Displaying marginal effect at the mean (MEM): \n')
            self.t_stat = self.mem/self.se_mem
            self.pval = 2*(t_dist.cdf(-abs(self.t_stat), self.n-self.k))
            nstars = np.zeros(len(self.pval))
            nstars[ self.pval <= .01 ] = 3
            nstars[ (self.pval > .01) * (self.pval <= .05) ] = 2
            nstars[ (self.pval > .05) * (self.pval <= .10) ] = 1
            stars = ['*'*int(i) for i in nstars]
            #
            output = pd.DataFrame({'Var.':self.vars,
                                'MEM':self.mem,
                                'Std.Err.':self.se_mem,
                                't-Stat.':self.t_stat,
                                'p-Val.': self.pval,
                                'Stars': stars
                                })

        output = output.round(decimals=digits)
        return output
    



    
    def mfx(self):
        """ Compute marginal effects and SEs. """
        #
        ## AME:
        latent = self.x@self.beta
        F = self.F_logit(latent)
        f = F*(1-F)
        f_bar = np.mean(f)
        f_p = F*(1-F)*(1-2*F)
        self.ame = f_bar * self.beta
        Gp = f_bar*np.eye(self.k) + ( self.x.T @ np.tile( f_p,(self.k,1) ).T @ np.diag(self.beta)/self.n ).T
        #Gp = f_bar*np.eye(self.k) + self.x.T @ np.diag( f_p ) @ np.tile(self.beta, (self.n,1))/self.n
        # sum = np.zeros((self.k,self.k))
        # for i in range(self.n):
        #     sum =  sum + f_p[i]*np.outer(self.beta, self.x[i,:])
        # Gp = f_bar*np.eye(self.k) + sum/self.n
        vcv = Gp @ self.vcv @ Gp.T
        self.se_ame = np.sqrt( np.diag(vcv))
        #
        ## MEM:       
        x_bar = np.mean(self.x,axis=0)
        latent_bar = x_bar @ self.beta
        F = self.F_logit(latent_bar)
        f = F*(1-F)
        self.mem = f*self.beta
        f_p = F*(1-F)*(1-2*F)
        Gp = f*np.eye(self.k) + f_p * np.outer(self.beta, x_bar)
        vcv = Gp @ self.vcv @ Gp.T
        self.se_mem = np.sqrt( np.diag(vcv))
        


    def run(self,se_type = '',display='mem'):
        """ Run regression and print results. """
        self.solve_gd()
        self.compute_se(se_type)
        self.mfx()
        self.output = self.summary(display=display)
        #
        with open('log.txt','w') as f:
            print(self.output,file=f)
        #
        print('\n', self.output, '\n')



