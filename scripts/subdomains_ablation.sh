for n in 10 20 50 100 200 500; do
    for seed in 11 12 13 14 15; do
        python experiments/allen_cahn_1d/allen_cahn_1d_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
        python experiments/burgers1d/burgers1d_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
    done
done

for n in 5 10 20 40 50 100; do
    for seed in 11 12 13 14 15; do
        python3 experiments/wave2d_marmousi/wave2d_marmousi_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
        python3 experiments/wave2d_three_layers/wave2d_three_layers_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
    done
done

for n in 100 200 500 1000 2000 5000; do
    for seed in 12 13 14 15 16; do
        python experiments/allen_cahn_2d/allen_cahn_2d_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed 
    done
done

for n in 5 10 20 50 60 150; do
    for seed in 11 12 13 14 15; do
        python3 experiments/wave3d_utah_forge/wave3d_utah_forge_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
    done
done