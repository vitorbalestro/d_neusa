import json
import numpy as np
import pandas as pd

models = ["NeuSA","D_NeuSA"]
metrics = ['relative_l2_error', 'training_time']
equations = ["allen_cahn_1d","burgers1d","allen_cahn_2d"]

seeds = [11, 12, 14, 15, 17]
dir_name = "baseline"

rows = []

for equation in equations:
    for model in models:
        for metric in metrics:
            metric_values = []
            for seed in seeds:
                path = (
                    f"results/{equation}/{dir_name}/{model}/seed_{seed}/metrics/metrics.json"     
                )
                try:
                    if equation == "allen_cahn_2d" and model == "NeuSA":
                        metric_values.append(0.0)
                    else:    
                        with open(path, "r", encoding="utf-8") as f:
                            dados = json.load(f)
                            metric_values.append(dados[metric])
                except Exception as e:
                    print(f"Error on opening {path}: {e}")
                    continue

            if metric_values:
                rows.append({
                    "equation": equation,
                    "model": model,
                    "metric": metric,
                    "mean": round(np.mean(metric_values), 4),
                })

df = pd.DataFrame(rows)

col_names_dict = {"relative_l2_error": "rMSE", "relative_l1_error": "rMAE", "training_time": "TT"}

mean_df = df.pivot(index=["equation", "model"], columns="metric", values="mean")
mean_df.columns = [f"{col_names_dict[col]} (mean)" for col in mean_df.columns]

ordered_columns = [
    "rMSE (mean)",
    "TT (mean)"
]

final_df = pd.concat([mean_df], axis=1)

final_df = final_df[ordered_columns]

with open("paper_tables/quantitative_comparison.txt", "w", encoding="utf-8") as f:
    f.write(final_df.to_string(float_format="%.6f"))