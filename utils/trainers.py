import torch.optim as optim
import torch
import os

class AdamTrainer:
    def __init__(
            self,
            params,
            adam_lr,
            adam_steps, 
            learning_rate_scheduler, 
            loss_function, 
            adam_eps=1e-8, 
            adam_amsgrad=False,
            model=None,
            checkpoint_steps=None,
            checkpoints_dir=None,
        ):

        self.params = params
        self.loss_function = loss_function

        self.adam_steps = adam_steps
        self.adam_lr = adam_lr
        self.adam_eps = adam_eps
        self.adam_amsgrad = adam_amsgrad

        self.optimizer = optim.Adam(self.params, lr=self.adam_lr, eps=self.adam_eps, amsgrad=adam_amsgrad)

        self.learning_rate_scheduler = learning_rate_scheduler
        self.lr_scheduler_mlp = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.lr_scheduler_neusa = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999, last_epoch=-1)

        self.lr_track = []
        self.loss_track_total = []
        self.loss_track_ic = []
        self.loss_track_bc = []
        self.loss_track_res = []

        self.checkpoint_steps = checkpoint_steps
        self.checkpoints_dir = checkpoints_dir
        self.model = model

        if self.checkpoint_steps:
            if self.model is None:
                raise ValueError(
                    "model must be provided when using checkpoints."
                )

            if self.checkpoints_dir is None:
                raise ValueError(
                    "checkpoint_dir must be provided when using checkpoints."
                )

            os.makedirs(self.checkpoints_dir, exist_ok=True)

    
    def train(self):
        for step in range(self.adam_steps):
            self.optimizer.zero_grad()

            losses = self.loss_function()
            loss_ic, loss_bc, loss_res = losses
            total_loss = sum(losses)
            
            total_loss.backward()
            self.optimizer.step()

            if self.learning_rate_scheduler and step % 100 == 0 and step != 0:
                self.lr_scheduler_mlp.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_track.append(current_lr)

            self.loss_track_total.append(total_loss.item())
            self.loss_track_ic.append(loss_ic.item())
            self.loss_track_bc.append(loss_bc.item())
            self.loss_track_res.append(loss_res.item())

            current_step = step + 1

            if self.checkpoint_steps is not None and current_step in self.checkpoint_steps:

                checkpoint_path = os.path.join(
                    self.checkpoints_dir,
                    f"model_step_{current_step}.pt"
                )

                torch.save(
                    self.model.state_dict(),
                    checkpoint_path
                )

                print(
                    f"Checkpoint saved at step {current_step}: "
                    f"{checkpoint_path}"
                )
            if step % 10 == 0 or step == self.adam_steps - 1:
                print(f"Adam step {step} | Loss: {total_loss.item():.6f}")
        print(f"Adam Optimization Completed. ")
        print(f'Adam training loss: {total_loss.item():.6f}')

    def train_neusa(self):
        for step in range(self.adam_steps):
            self.optimizer.zero_grad()

            loss = self.loss_function()

            if step == 0:
                self.initial_loss = loss.detach()

            loss.backward()

            self.optimizer.step()

            self.lr_scheduler_neusa.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_track.append(current_lr)

            self.loss_track_total.append(loss.item())
            if step % 10 == 0 or step == self.adam_steps - 1:
                print(f"Adam step {step} | Loss: {loss.item():.12f}")
        print(f"Adam Optimization Completed. ")
        self.final_loss = loss.detach()
        print(f'Adam training loss: {self.final_loss.item():.12f}')
    
    def train_neusa_best_checkpoint(self):
        best_loss = float("inf")
        best_state = None
        for step in range(self.adam_steps):
            self.optimizer.zero_grad()

            loss = self.loss_function()

            loss.backward()

            self.optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = [p.clone().detach() for p in self.params]

            self.lr_scheduler_neusa.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_track.append(current_lr)

            self.loss_track_total.append(loss.item())
            if step % 10 == 0 or step == self.adam_steps - 1:
                print(f"Adam step {step} | Loss: {loss.item():.12f}")
            
        print(f"Adam Optimization Completed. ")
        print(f'Adam best training loss: {best_loss:.12f}')
        with torch.no_grad():
            for p, best_p in zip(self.params,best_state):
                p.copy_(best_p)

    
class AdamTrainerWithWeights:
    def __init__(self,params,adam_lr, adam_steps, learning_rate_scheduler, loss_function, adam_eps=1e-8, adam_amsgrad=False, weight_scale=1.0):

        self.params = params
        self.loss_function = loss_function

        self.adam_steps = adam_steps
        self.adam_lr = adam_lr
        self.adam_eps = adam_eps
        self.adam_amsgrad = adam_amsgrad

        self.optimizer = optim.Adam(self.params, lr=self.adam_lr, eps=self.adam_eps, amsgrad=adam_amsgrad)

        self.learning_rate_scheduler = learning_rate_scheduler
        self.lr_scheduler_mlp = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.lr_scheduler_neusa = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999, last_epoch=-1)

        self.weight_scale = weight_scale
        self.lr_track = []
        self.loss_track_total = []

    def train_neusa(self,W):
        
        for step in range(self.adam_steps):
            self.optimizer.zero_grad()

            loss,h_lhs,h_rhs = self.loss_function()

            loss.backward()

            self.optimizer.step()

            self.lr_scheduler_neusa.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_track.append(current_lr)

            self.loss_track_total.append(loss.item())
            if step % 10 == 0 or step == self.adam_steps - 1:
                print(f"Adam step {step} | Loss: {loss.item():.12f}")
            
            with torch.no_grad():
                total = torch.sum((h_lhs-h_rhs)**2)
                for k in range(h_lhs.shape[1]):
                    mean_error = torch.sum((h_lhs[:,k]-h_rhs[:,k])**2) / total
                    W[:,k] += self.weight_scale * (1+step/100) * mean_error
        print(f"Adam Optimization Completed. ")
        print(f'Adam training loss: {loss.item():.12f}')
        

    def train_neusa_best_checkpoint(self,W):
        best_loss = float("inf")
        best_state = None
        for step in range(self.adam_steps):
            self.optimizer.zero_grad()

            loss,hv_t,h_rhs = self.loss_function()

            loss.backward()

            self.optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = [p.clone().detach() for p in self.params]

            self.lr_scheduler_neusa.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_track.append(current_lr)

            self.loss_track_total.append(loss.item())
            if step % 10 == 0 or step == self.adam_steps - 1:
                print(f"Adam step {step} | Loss: {loss.item():.12f}")
            
            with torch.no_grad():
                total = torch.sum((hv_t-h_rhs)**2)
                for k in range(hv_t.shape[1]):
                    mean_error = torch.sum((hv_t[:,k]-h_rhs[:,k])**2) / total
                    W[:,k] += self.weight_scale * mean_error
        print(f"Adam Optimization Completed. ")
        print(f'Adam best training loss: {best_loss:.12f}')
        with torch.no_grad():
            for p, best_p in zip(self.params,best_state):
                p.copy_(best_p)

