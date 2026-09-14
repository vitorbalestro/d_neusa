for diffusion in 0.1 0.01 0.001 0.0001 0.00001 0.000001; do
    for seed in 11 12 13 14 15; do
        python experiments/allen_cahn_1d/allen_cahn_1d_D_NeuSA.py --diffusion $diffusion --seed $seed --dir ablation_diffusion_coefficient --ablation_diffusion
        python experiments/allen_cahn_1d/allen_cahn_1d_D_NeuSA.py --n_slices 1 --diffusion $diffusion --seed $seed --dir ablation_diffusion_coefficient --ablation_diffusion
    done
done


for seed in 11 12 13 14 15; do
    python experiments/burgers1d/burgers1d_D_NeuSA.py --ablation_diffusion --dir ablation_diffusion_coefficient --seed $seed --diffusion 0.01 --discretization_size 5001
done

for seed in 11 12 13 14 15; do
    python experiments/burgers1d/burgers1d_D_NeuSA.py --ablation_diffusion --dir ablation_diffusion_coefficient --seed $seed --diffusion 0.1 --discretization_size 30001
done

for seed in 11 12 13 14 15; do
        python experiments/burgers1d/burgers1d_D_NeuSA.py --ablation_diffusion --dir ablation_diffusion_coefficient --seed $seed --diffusion 0.001 --discretization_size 30001 --widen_factor 4
done