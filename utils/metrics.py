import numpy as np

class Metrics:
    def __init__(self,u_pred,u_exact):
        self.u_pred = u_pred
        self.u_exact = u_exact
    
    def relative_l2_error(self):
        return np.sqrt(np.sum((self.u_pred-self.u_exact)**2)/np.sum((self.u_exact)**2))

    def relative_l1_error(self):
        return np.sum(np.abs(self.u_pred-self.u_exact))/np.sum(np.abs(self.u_exact))
    