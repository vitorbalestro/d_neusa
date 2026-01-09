# Neuro-Spectral Architectures With Time Domain Decomposition

# Setup and Dependencies

The dependencies are managed with Poetry (https://python-poetry.org/). If you don't have Poetry installed yet:
```sh
pip install poetry
```
Then, from the project root:
```sh
poetry install
```
This command will create a virtual environment and install all dependencies listed in `pyproject.toml`.

After installation, you can activate poetry shell (also from the project root):
```sh
poetry shell
```
and run each `.py` file as usual.

If you prefer not to use Poetry, you can install the dependencies listed in `pyproject.toml` manually.

# Training and Evaluating

We implemented NeuSA and D-NeuSA models for seven equations: Allen-Cahn, Kuramoto-Sivashinsky, Klein-Gordon non-linear with Gaussian initial condition, Burgers 1d, Wave 2d with a three layered medium, Wave 2d with a Marmousi medium and Wave 3d with the Utah FORGE medium.

Each equation has two files. One of them implements the NeuSA method, and the other one implements the D-NeuSA method.

When you run one of these scripts, the corresponding model is trained and compared with a ground-truth solution. Some plots are produced as well. The comparison metrics, the trained model and the plots are automatically saved in a subfolder of the `results` directory.

At each script, an argparse provides the following options:

--dir: the name of the subfolder of 'results' directory where the results will be stored;<br>
--seed: the chosen random torch seed;<br>
--device: the device where the code will be executed (a GPU, preferably);<br>
--steps: the number of training steps (we adopted an Adam trainer for all experiments);<br>
--lr: the Adam trainer learning rate;<br>

The default values for the learning rate and for the number of training steps in each script are the ones that we have used to produce the numbers that appear in the paper.

For example, if you run

```sh
python wave2d_marmousi_D_NeuSA.py --dir my_experiment
```

a D-NeuSA model for the Wave 2d Marmousi equation will be trained for 2000 steps (each submodel) with learning rate 0.01, since these are the default values in the argparse. The comparison metrics, the trained model and the plots will be saved in the folder `/results/wave2d_marmousi/my_experiment/D_NeuSA/seed_42`, because 42 is the default seed in the argparse. 
