for seed in 11 12 14 15 17; do
    python allen_cahn_1d_D_NeuSA.py --seed $seed --dir baseline
    python allen_cahn_1d_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --steps 200

    python burgers1d_D_NeuSA.py --seed $seed --dir baseline
    python burgers1d_D_NeuSA.py --seed $seed --dir baseline --n_slices 1 --steps 200

    python allen_cahn_2d_D_NeuSA.py --seed $seed --dir baseline
    python allen_cahn_2d_D_NeuSA --seed $seed --dir baseline --n_slices 1 --discretization_size 1001
done