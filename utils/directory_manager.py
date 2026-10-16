import os

class DirectoryManager:
    def __init__(
        self,
        equation,
        model,
        dir_name,
        seed,
        ablation=False,
        specifier=None,
        checkpoints=False,
        subdomain_losses=False
    ):
        self.equation = equation
        self.ablation = ablation
        self.specifier = specifier
        self.seed = seed
        self.model = model
        self.dir_name = dir_name
        self.checkpoints = checkpoints
        self.subdomain_losses = subdomain_losses
        
    
    def config(self):

        print("Current working directory:", os.getcwd())

        if not self.ablation:
            self.save_dir = os.path.join("results",self.equation,self.dir_name,self.model,f"seed_{self.seed}")
        else:
            self.save_dir = os.path.join("results",self.equation,self.dir_name,self.specifier,self.model,f"seed_{self.seed}")
        
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_dir_solutions = os.path.join(
            self.save_dir, "solutions"
        )
        self.save_dir_metrics = os.path.join(
            self.save_dir, "metrics"
        )
        self.figs_dir = os.path.join(
            self.save_dir, "figures"
        )

        if self.checkpoints:
            self.checkpoints_dir = os.path.join(
                self.save_dir, "checkpoints"
            )
            os.makedirs(self.checkpoints_dir, exist_ok=True)

        if self.subdomain_losses:
            self.losses_dir = os.path.join(
                self.save_dir, "losses"
            )
            os.makedirs(self.losses_dir, exist_ok=True)

        os.makedirs(self.save_dir_solutions, exist_ok=True)
        os.makedirs(self.save_dir_metrics, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)