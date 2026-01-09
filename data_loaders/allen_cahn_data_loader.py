import os
import numpy as np
import scipy.io
from utils.grid import Grid2DToTensors

class AllenCahn1DGroundTruthLoader:
    def __init__(self):
        
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "allen_cahn", "AC.mat")
        data = scipy.io.loadmat(file_path)

        self.t = data['tt'].flatten()[:,None]
        self.x = data['x'].flatten()[:,None]
        self.u = np.real(data['uu']).T

    def get_ground_truth_grid(self):
        return self.x, self.t 
    
    def get_ground_truth_solution(self):
        return self.u 
    
    def get_ground_truth_grid_as_tensors(self):

        grid_ground_truth = Grid2DToTensors(self.x, self.t)
        x_tensor = grid_ground_truth.grid_as_column_x
        t_tensor = grid_ground_truth.grid_as_column_t
        
        return x_tensor, t_tensor
    
    def get_grid_size(self):
        return len(self.x), len(self.t)

