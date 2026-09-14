for seed in 11 12 14 15 17; do
    python experiments/allen_cahn_1d/allen_cahn_1d_D_NeuSA.py --seed $seed --dir baseline
    python experiments/allen_cahn_1d/allen_cahn_1d_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --steps 200

    for model in PINN QRes FLS; do
        python experiments/allen_cahn_1d/allen_cahn_1d_mlp_models.py --seed $seed --dir baseline --model $model
        python experiments/allen_cahn_1d/allen_cahn_1d_mlp_models.py --seed $seed --dir baseline --causal --model $model
    done

    python experiments/burgers1d/burgers1d_D_NeuSA.py --seed $seed --dir baseline
    python experiments/burgers1d/burgers1d_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --steps 200

    for model in PINN QRes FLS; do
        python experiments/burgers1d/burgers1d_mlp_models.py --seed $seed --dir baseline --model $model
        python experiments/burgers1d/burgers1d_mlp_models.py --seed $seed --dir baseline --causal --model $model --epsilon 1
    done

    python experiments/allen_cahn_2d/allen_cahn_2d_D_NeuSA.py --seed $seed --dir baseline
    python experiments/allen_cahn_2d/allen_cahn_2d_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --discretization_size 1001

    for model in PINN QRes FLS; do
        python experiments/allen_cahn_2d/allen_cahn_2d_mlp_models.py --seed $seed --dir baseline --model $model
        python experiments/allen_cahn_2d/allen_cahn_2d_mlp_models.py --seed $seed --dir baseline --causal --model $model
    done

    python experiments/wave2d_three_layers/wave2d_three_layers_D_NeuSA.py --seed $seed --dir baseline
    python experiments/wave2d_three_layers/wave2d_three_layers_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --steps 1000

    for model in PINN QRes FLS; do
        python experiments/wave2d_three_layers/wave2d_three_layers_mlp_models.py --seed $seed --dir baseline --model $model
        python experiments/wave2d_three_layers/wave2d_three_layers_mlp_models.py --seed $seed --dir baseline --causal --model $model
    done

    python experiments/wave2d_marmousi/wave2d_marmousi_D_NeuSA.py --seed $seed --dir baseline
    python experiments/wave2d_marmousi/wave2d_marmousi_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --steps 3000

    for model in PINN QRes FLS; do
        python experiments/wave2d_marmousi/wave2d_marmousi_mlp_models.py --seed $seed --dir baseline --model $model
        python experiments/wave2d_marmousi/wave2d_marmousi_mlp_models.py --seed $seed --dir baseline --causal --model $model
    done

    python experiments/wave3d_utah_forge/wave3d_utah_forge_D_NeuSA.py --seed $seed --dir baseline 
    python experiments/wave3d_utah_forge/wave3d_utah_forge_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --steps 5000

    for model in PINN QRes FLS; do
        python experiments/wave3d_utah_forge/wave3d_utah_forge_mlp_models.py --seed $seed --dir baseline --model $model
        python experiments/wave3d_utah_forge/wave3d_utah_forge_mlp_models.py --seed $seed --dir baseline --causal --model $model
    done
done