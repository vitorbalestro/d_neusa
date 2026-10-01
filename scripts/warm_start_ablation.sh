for seed in 11 12 14 15 17; do

    python experiments/allen_cahn_1d/allen_cahn_1d_D_NeuSA.py --seed $seed --dir warm_start_ablation --warm_start
    python experiments/burgers1d/burgers1d_D_NeuSA.py --seed $seed --dir warm_start_ablation --warm_start
    python experiments/allen_cahn_2d/allen_cahn_2d_D_NeuSA.py --seed $seed --dir warm_start_ablation --warm_start
    python experiments/wave2d_three_layers/wave2d_three_layers_D_NeuSA.py --seed $seed --dir warm_start_ablation --warm_start
    python experiments/wave2d_marmousi/wave2d_marmousi_D_NeuSA.py --seed $seed --dir warm_start_ablation --warm_start
    python experiments/wave3d_utah_forge/wave3d_utah_forge_D_NeuSA.py --seed $seed --dir warm_start_ablation --warm_start

done