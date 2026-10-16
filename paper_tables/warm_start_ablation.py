import json
import numpy as np
import pandas as pd

models = ["D_NeuSA"]

equations = [
    "allen_cahn_1d",
    "allen_cahn_2d",
    "burgers1d",
    "wave2d_marmousi",
    "wave2d_three_layers",
    "wave3d_utah_forge"
]

seeds = [11, 12, 14, 15, 17]
dir_name = "warm_start_ablation"

rows = []

for equation in equations:
    for model in models:

        errors = []
        times = []

        for seed in seeds:
            path = (
                f"results/{equation}/{dir_name}/{model}/"
                f"seed_{seed}/metrics/metrics.json"
            )

            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                errors.append(data["relative_l2_error"])
                times.append(data["training_time"])

            except Exception as e:
                print(f"Error opening {path}: {e}")

        if errors:
            rows.append({
                "equation": equation,
                "model": model,
                "rRMSE": np.mean(errors),
                "TT": np.mean(times),
            })

df = pd.DataFrame(rows)

model_names = {
    "D_NeuSA":"D_NeuSA",
}

equation_names = {
    "allen_cahn_1d": "Allen--Cahn 1D",
    "allen_cahn_2d": "Allen--Cahn 2D",
    "burgers1d": "Burgers",
    "wave2d_marmousi": "Marmousi",
    "wave2d_three_layers": "Three Layers",
    "wave3d_utah_forge": "Utah FORGE"
}

df["model"] = df["model"].map(model_names)
df["equation"] = df["equation"].map(equation_names)

# Put problems in columns
table = df.pivot(
    index="model",
    columns="equation",
    values=["rRMSE", "TT"]
)

# Desired order
models_order = ["D_NeuSA"]
equations_order = ["Allen--Cahn 1D","Allen--Cahn 2D", "Burgers", "Three Layers", "Marmousi", "Utah FORGE"]

with open(
    "paper_tables/warm_start_ablation.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"{'Model':<10}"
        f"{'Allen--Cahn 1D':>28}"
        f"{'Allen--Cahn 2D':>28}"
        f"{'Burgers':>28}"
        f"{'Three Layers':>28}"
        f"{'Marmousi':>28}"
        f"{'Utah FORGE':>28}\n"
    )

    f.write(
        f"{'':<10}"
        f"{'rRMSE':>14}{'TT':>14}" 
        f"{'rRMSE':>14}{'TT':>14}"
        f"{'rRMSE':>14}{'TT':>14}"
        f"{'rRMSE':>14}{'TT':>14}"
        f"{'rRMSE':>14}{'TT':>14}"
        f"{'rRMSE':>14}{'TT':>14}\n"
    )

    for model in models_order:

        line = f"{model:<10}"

        for equation in equations_order:
            rrmse = table.loc[model, ("rRMSE", equation)]
            tt = table.loc[model, ("TT", equation)]

            line += f"{rrmse:>14.6f}{tt:>14.2f}"

        f.write(line + "\n")

print(open("paper_tables/warm_start_ablation.txt").read())