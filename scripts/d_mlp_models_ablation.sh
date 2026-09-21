for seed in 11 12 14 15 17; do
    for model in PINN QRes FLS; do
        python experiments/allen_cahn_1d/allen_cahn_1d_D_mlp_models.py --dir d_mlp_ablation --seed $seed --model $model
        python experiments/wave2d_marmousi/wave2d_marmousi_D_mlp_models.py --dir d_mlp_ablation --seed $seed --model $model
        python experiments/burgers1d/burgers1d_D_mlp_models.py --dir d_mlp_ablation --seed $seed --model $model
    done
done