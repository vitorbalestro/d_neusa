import time
import torch
from torchdyn.core import NeuralODE

from utils.trainers import AdamTrainer
from models.mlps import MLP, MLPWithTranspose2D, MLPWithTranspose3D

class TrainingWrapperDNeuSA:
    def __init__(
        self,
        neural_network_dict,
        optim_dict,
        ts,
        n_slices,
        hu0,
        basis,
        F0,
        neural_vector_field_builder,    # HeatLikeVectorField, e.g.
        model_weight,
        loss_function,
        space_dimension=1,
        loss_kwargs=None,
        vector_field_kwargs=None,
        decompose_output=False,
        device='cpu',
        warm_start=False
    ):

        self.device = torch.device(device)

        self.neural_network_dict = neural_network_dict
        self.optim_dict = optim_dict
        self.ts = ts.to(self.device)
        self.n_slices = n_slices

        self.n_t = len(self.ts)
        assert (self.n_t-1) % self.n_slices == 0
        self.offset = (self.n_t - 1) // self.n_slices

        self.time_slices = [self.offset * i for i in range(1,n_slices+1)]

        self.hu0 = hu0.to(device)
        self.basis = basis.to(device)
        self.F0 = F0.to(device)
        self.neural_vector_field_builder = neural_vector_field_builder
        self.model_weight = model_weight

        self.space_dimension = space_dimension

        self.loss_function = loss_function
        self.loss_kwargs = {} if loss_kwargs is None else loss_kwargs.copy()
        self.vector_field_kwargs = {} if vector_field_kwargs is None else vector_field_kwargs

        self.decompose_output=decompose_output

        self.inference_time = 0.0

        self.warm_start = warm_start

    def initialize_subnetwork(self):

        if self.space_dimension == 1:
            F_neural = MLP(
                self.neural_network_dict['in_dim'],
                self.neural_network_dict['hidden_dim'],
                self.neural_network_dict['out_dim'],
                self.neural_network_dict['num_hidden_layers']
            )
            
        elif self.space_dimension == 2:
            F_neural = MLPWithTranspose2D(
                M = self.neural_network_dict['M'],
                N = self.neural_network_dict['N'],
                widen_factor = self.neural_network_dict['widen_factor'],
                num_hidden_layers = self.neural_network_dict['num_hidden_layers']
            )

        elif self.space_dimension == 3:
            F_neural = MLPWithTranspose3D(
                M = self.neural_network_dict['M'],
                N = self.neural_network_dict['N'],
                P = self.neural_network_dict['P'],
                widen_factor = self.neural_network_dict['widen_factor'],
                num_hidden_layers = self.neural_network_dict['num_hidden_layers']
            )

        F = self.neural_vector_field_builder(
            F_neural,
            self.F0,
            model_weight=self.model_weight,
            **self.vector_field_kwargs
        )


        NODE = NeuralODE(
            F,
            solver='rk4',
            sensitivity='autograd',
            return_t_eval=False
        )

        F = F.to(self.device)
        NODE = NODE.to(self.device)

        return F_neural,F,NODE
    
    def count_parameters(self,NODE):
        num_trainable_params = self.n_slices * sum(
            p.numel() for p in NODE.parameters() if p.requires_grad
        )

        self.num_params = num_trainable_params

    def train(self):

        hus = []

        self.start_time = time.time()

        F_neural,F, NODE = self.initialize_subnetwork()

        trainer = AdamTrainer(
            NODE.parameters(),
            adam_lr=self.optim_dict['adam_lr'],
            adam_steps=self.optim_dict['steps'],
            learning_rate_scheduler=self.optim_dict['learning_rate_scheduler'],
            loss_function = lambda: self.loss_function(
                hu0=self.hu0,
                ts=self.ts[0:self.time_slices[0]+1],
                NODE=NODE,
                basis=self.basis,
                F=F,
                F0=self.F0,
                **self.loss_kwargs
            )
        )

        trainer.train_neusa()

        with torch.no_grad():
            start_inf_time = time.time()
            hu = NODE(self.hu0,self.ts[0:self.time_slices[0]+1])
            end_inf_time = time.time()
        
        self.inference_time += end_inf_time - start_inf_time

        hus.append(hu.cpu())

        # here we compute and store the number of parameters
        self.count_parameters(NODE)

        if not self.warm_start:
            del NODE
            del F
            del F_neural

        del trainer
        del hu

        for i in range(1,self.n_slices):
            
            if not self.warm_start:
                F_neural,F, NODE = self.initialize_subnetwork()

            hu_init = hus[i-1][-1].detach().to(self.device)
            hus[i-1] = hus[i-1][:-1]

            trainer = AdamTrainer(
                NODE.parameters(),
                adam_lr=self.optim_dict['adam_lr'],
                adam_steps=self.optim_dict['steps'],
                learning_rate_scheduler=self.optim_dict['learning_rate_scheduler'],
                loss_function = lambda: self.loss_function(
                    hu0=hu_init,
                    ts=self.ts[self.time_slices[i-1]:self.time_slices[i]+1],
                    NODE=NODE,
                    basis=self.basis,
                    F=F,
                    F0=self.F0,
                    **self.loss_kwargs
                )
            )

            trainer.train_neusa()

            with torch.no_grad():
                start_inf_time = time.time()
                hu = NODE(hu_init,self.ts[self.time_slices[i-1]:self.time_slices[i]+1])
                end_inf_time = time.time()
            
            self.inference_time += end_inf_time - start_inf_time

            hus.append(hu.cpu())

            if not self.warm_start:
                del NODE
                del F
                del F_neural

            del trainer
            del hu

            
        self.end_time = time.time()
        
        self.training_time = self.end_time - self.start_time

        hu = torch.cat(hus,dim=0).to(self.device)
        if not self.decompose_output:
            u_pred = self.basis.idbt(hu).squeeze().detach().cpu().numpy()
        else:
            hu,_ = hu.chunk(2,dim=-1)
            u_pred = self.basis.idbt(hu).squeeze().detach().cpu().numpy()

        return u_pred