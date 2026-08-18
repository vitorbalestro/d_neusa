for n in 10 20 50 100 200 500; do
    for seed in 11 12 13 14 15; do
        python allen_cahn_1d_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
        python burgers1d_D_NeuSA.py --ablation_subdomains --dir ablation_subdomains --n_slices $n --seed $seed
    done
done