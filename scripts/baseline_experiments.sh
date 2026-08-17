for seed in 11 12 14 15 17; do
    python allen_cahn_1d_D_NeuSA.py --seed $seed --dir baseline
    python allen_cahn_1d_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --steps 1000
done