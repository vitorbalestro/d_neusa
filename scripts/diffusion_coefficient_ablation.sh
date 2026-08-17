for diffusion in 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
    for seed in 11 12 13 14 15; do
        python allen_cahn_1d_D_NeuSA.py --diffusion $diffusion --seed $seed --dir ablation_diffusion_coefficient --ablation_diffusion
        python allen_cahn_1d_D_NeuSA.py --n_slices 1 --diffusion $diffusion --seed $seed --dir ablation_diffusion_coefficient --ablation_diffusion 
    done
done