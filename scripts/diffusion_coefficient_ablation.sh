for diffusion in 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
    for seed in 11 12 13 14 15; do
        python allen_cahn_1d_D_NeuSA.py --diffusion $diffusion --seed $seed --dir ablation_diffusion_coefficient --ablation_diffusion
        python allen_cahn_1d_D_NeuSA.py --n_slices 1 --diffusion $diffusion --seed $seed --dir ablation_diffusion_coefficient --ablation_diffusion
    done
done

for diffusion in 0.1 0.001; do
    for seed in 11 12 13 14 15; do
        python burgers1d_D_NeuSA.py --ablation_diffusion --dir ablation_diffusion_coefficient --seed $seed --diffusion $diffusion --discretization_size 30001
    done
done

for diffusion in 0.01; do
    for seed in 11 12 13 14 15; do
        python burgers1d_D_NeuSA.py --ablation_diffusion --dir ablation_diffusion_coefficient --seed $seed --diffusion $diffusion --discretization_size 5001
    done
done