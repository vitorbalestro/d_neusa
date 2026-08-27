for n in 10 20 50 100 200 500; do
    for seed in 11 12 13 14 15; do
        python allen_cahn_1d_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
        python burgers1d_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
    done
done

for n in 5 10 20 40 50 100; do
    for seed in 10 11 13 14 15; do
        python3 wave2d_marmousi_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
        python3 wave2d_three_layers_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
    done
done

for n in 100 200 500 1000 2000 5000; do
    for seed in 11 12 13 14 15; do
        python allen_cahn_2d_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed 
    done
done

for n in 5 10 20 50 60 150; do
    for seed in 11 12 13 14 15; do
        python3 wave3d_utah_forge_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
    done
done