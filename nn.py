import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NN:
    """ Nearest neighbors classification/regression. """
    def __init__(self,x,y,k=None):
        """ Initialize kNN. """
        ## Decide regression or classification.

        ## Filter missings.
        check_x = np.isnan(x)
        check_y = np.isnan(y)
        keep_row = ( (np.sum(check_x,axis=1)+check_y) == 0)
        NAs = np.sum(keep_row)
        n = len(y)
        if NAs<len(y) :
            print(f'Due to missing values, {NAs} observations of {n} were dropped.')
            self.x = x.loc[keep_row,:]
            self.y = y[keep_row]
        else:
            self.x = x
            self.y = y
        #
        self.n = len(self.y)
        self.l = x.shape[1]
        self.k = int(np.floor(np.sqrt(self.x.shape[0])))
        #
        self.residuals = None
        self.sse = None
        self.mse = None
        self.rmse = None
        self.ser = None
        self.rsq = None
        #
        self.x_min = self.x.min().tolist()
        self.x_max = self.x.max().tolist()
        self.u = self.x.apply(self.maxmin_norm)
        #
        self.y_hat = None
        self.y_new = None
        # If k is none, pick k by cross validation

    def maxmin_norm(self,x):
        """ Max-min normalize a variable. """
        x_max = max(x)
        x_min = min(x)
        u = (x-x_min)/(x_max-x_min)
        return u
    
    def create_folds(self,n,n_folds):
        index = np.linspace(0,n-1,n) # Index all the rows
        np.random.shuffle(index) # Randomize the order of the rows
        fold_size = int(np.floor(n/n_folds))
        max_index = fold_size*n_folds
        cv_folds = np.reshape(index[0:max_index],(fold_size,n_folds)) # Create K folds of N/K values each
        cv_grid = range(n_folds)
        return cv_folds, cv_grid

        
    def sq_dist(self,X,Z):
        """ Compute squared distance matrix from X to Z. """
        X = X.to_numpy()
        Z = Z.to_numpy()
        Xsq = np.sum(X**2,axis=1)
        Zsq = np.sum( Z**2,axis=1).T
        X_mat = np.tile(Xsq,(Zsq.shape[0],1)).T
        Z_mat = np.tile(Zsq,(Xsq.shape[0],1))
        d = X_mat - 2*(X@Z.T)  + Z_mat
        return d


    def k_by_cv_reg(self,k_max=None, n_folds =10, k_steps = None):
        """ Pick K by cross validation on training data. """
        k_min = 3
        if k_max is None:
            k_max = min( 2*int( np.floor( np.sqrt( self.x.shape[0] ) ) ), self.n )
        if k_steps is None:
            k_steps = int(np.floor(np.sqrt(k_max - k_min)))
            print('Number of steps for k: ', k_steps)
            print('Max k: ', k_max)
            print('Min k: ', k_min)
            print('Number of folds: ', n_folds)
        k_nn_grid = np.arange(k_min, k_max, k_steps)
        L_nn = len(k_nn_grid)
        ## Create folds:
        folds, cv_grid = self.create_folds(self.n,n_folds)
        ## Compute RMSE for fold k:
        score_rmse = np.reshape(np.zeros( len(k_nn_grid)*len(cv_grid) ),(len(k_nn_grid),len(cv_grid)))
        ## Cross-validate
        for k_nn_index in range(L_nn):
            k_nn = k_nn_grid[k_nn_index]
            for k_cv in cv_grid:
                # Get data for this fold:
                u_nk = self.u.drop(folds[:,k_cv],axis=0)
                y_nk = self.y.drop(folds[:,k_cv],axis=0)
                u_k = self.u.iloc[folds[:,k_cv],:]
                y_k = self.y[folds[:,k_cv]]
                # Make prediction:
                y_k_hat = ( np.argsort(np.argsort( self.sq_dist(u_k,u_nk), axis=1 ) , axis=1)< (k_nn) ) @ y_nk/(k_nn)
                # Compute and sore RMSE:
                r_k = y_k_hat - y_k
                score_rmse[k_nn_index, k_cv] = np.sqrt(r_k@r_k/self.n)
        rmse = np.mean(score_rmse,axis=1)
        self.k = k_nn_grid[np.argmin(rmse)]


    def normalize_new(self,x_new):
        """ Normalize new values using training normalization parameters. """
        u_new = self.x.copy().astype(dtype='float64')
        for i in range(u_new.shape[1]):
            x_convert = self.x.iloc[:,i].astype(dtype='float64')
            u_new.iloc[:,i] = (x_convert-float(self.x_min[i]))/(float(self.x_max[i]) - float(self.x_min[i]))
        return u_new


    def predict_reg(self, x_new, k = None, y_test = None):
        """ Predict values for new cases. """
        if k is None:
            k = self.k
        # Transform x_new by way of x_train:
        u_new = self.normalize_new(x_new)
        # Regression prediction: 
        y_new = ( np.argsort(np.argsort( self.sq_dist(u_new,self.u), axis=1 ) , axis=1)< k ) @ self.y/k
        self.y_new = y_new
        # If training outcomes available:
        if not (y_test is None) :
            r_model = y_test - y_new
            r_mean = y_test - np.mean(self.y)
            self.sse = np.inner(r_model,r_model)
            self.rsq_test = 1 - self.sse/np.inner(r_mean,r_mean)
            self.rmse_test = np.sqrt( self.sse/len(y_test) )
        return y_new
