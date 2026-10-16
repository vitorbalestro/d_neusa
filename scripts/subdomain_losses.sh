THIS MIGHT NOT BE NEEDED!!!!
JUST USE THE LOSSES OF DIR baseline AND DIR warm_start_ablation, SEED 11.


python experiments/allen_cahn_1d/allen_cahn_1d_D_NeuSA.py --seed 11 --dir subdomain_losses_warm_start --warm_start
python experiments/allen_cahn_1d/allen_cahn_1d_D_NeuSA.py --seed 11 --dir subdomain_losses_independent 

python experiments/allen_cahn_2d/allen_cahn_2d_D_NeuSA.py --seed 11 --dir subdomain_losses_warm_start --warm_start
python experiments/allen_cahn_2d/allen_cahn_2d_D_NeuSA.py --seed 11 --dir subdomain_losses_independent

python experiments/burgers_1d/burgers_1d_D_NeuSA.py --seed 11 --dir subdomain_losses_warm_start --warm_start
python experiments/burgers_1d/burgers_1d_D_NeuSA.py --seed 11 --dir subdomain_losses_independent 

python experiments/wave2d_marmousi/wave2d_marmousi_D_NeuSA.py --seed 11 --dir subdomain_losses_warm_start --warm_start
python experiments/wave2d_marmousi/wave2d_marmousi_D_NeuSA.py --seed 11 --dir subdomain_losses_independent

python experiments/wave2d_three_layers/wave2d_three_layers_D_NeuSA.py --seed 11 --dir subdomain_losses_warm_start --warm_start
python experiments/wave2d_three_layers/wave2d_three_layers_D_NeuSA.py --seed 11 --dir subdomain_losses_independent

python experiments/wave3d_utah_forge/wave3d_utah_forge_D_NeuSA.py --seed 11 --dir subdomain_losses_warm_start --warm_start
python experiments/wave3d_utah_forge/wave3d_utah_forge_D_NeuSA.py --seed 11 --dir subdomain_losses_independent

