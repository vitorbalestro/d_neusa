import os

class DirectoryManager:
    def __init__(
        self,
        equation,
        model,
        dir_name,
        seed,
        ablation=False,
        specifier=None
    ):
        self.equation = equation
        self.ablation = ablation
        self.specifier = specifier
        self.seed = seed
        self.model = model
        self.dir_name = dir_name
        
    
    def config(self):
    
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

        os.makedirs(self.save_dir_solutions, exist_ok=True)
        os.makedirs(self.save_dir_metrics, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)