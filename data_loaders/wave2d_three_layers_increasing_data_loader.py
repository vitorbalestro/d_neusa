import os
import numpy as np

from utils.grid import Grid3DToTensors

class Wave2DThreeLayersIncreasingGroundTruthLoader:
    def __init__(self):
        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wave2d_three_layers_increasing")
        self.x_file_path = os.path.join(self.data_dir_path,"x.txt")
        self.x = np.loadtxt(self.x_file_path)

        self.y_file_path = os.path.join(self.data_dir_path,"y.txt")
        self.y = np.loadtxt(self.y_file_path)
    
        self.t_file_path = os.path.join(self.data_dir_path,"t.txt")
        self.t = np.loadtxt(self.t_file_path)
        
    def get_ground_truth_grid(self):
        return self.x,self.y,self.t

    def get_ground_truth_solution(self):
        self.u_file_path = os.path.join(self.data_dir_path,"u.txt")
        self.u = np.loadtxt(self.u_file_path)

        return self.u
    
    def get_ground_truth_grid_as_tensors(self):

       
        grid_as_tensors = Grid3DToTensors(self.x,self.y,self.t)

        x_tensor = grid_as_tensors.x
        y_tensor = grid_as_tensors.y
        t_tensor = grid_as_tensors.t

        return x_tensor, y_tensor, t_tensor
    
    def get_grid_size(self):
        return len(self.x), len(self.y), len(self.t)