import json

equations = ["allen_cahn_1d","allen_cahn_2d","burgers1d","wave2d_marmousi","wave2d_three_layers","wave3d_utah_forge"]
seeds = [11,12,14,15,17]
neusa_times = []
dneusa_times = []
solver_times = []

for equation in equations:
    neusa_time = 0.0
    dneusa_time = 0.0
    solver_time = 0.0
    for seed in seeds:
        if equation != "allen_cahn_2d":
            neusa_path = f"results/{equation}/baseline/NeuSA/seed_{seed}/metrics/metrics.json"
        dneusa_path = f"results/{equation}/baseline/D_NeuSA/seed_{seed}/metrics/metrics.json"

        if equation != "allen_cahn_2d":
            with open(neusa_path,"r") as f:
                neusa_data = json.load(f)
            neusa_time += neusa_data["model_inference_time"]

        with open(dneusa_path,"r") as f:
            dneusa_data = json.load(f)
        dneusa_time += dneusa_data["model_inference_time"]
        solver_time += dneusa_data["solver_inference_time"]

    if equation != "allen_cahn_2d":
        neusa_time = neusa_time / len(seeds)
    else: 
        neusa_time = 0.0
    dneusa_time = dneusa_time / len(seeds)
    solver_time = solver_time / len(seeds)

    neusa_times.append(neusa_time)
    dneusa_times.append(dneusa_time)
    solver_times.append(solver_time)

output_path = "paper_tables/inference_times.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for i, equation in enumerate(equations):
        line = (
            f"Equation: {equation} | "
            f"D_NeuSA: {dneusa_times[i]:.6f} | "
            f"NeuSA: {neusa_times[i]:.6f} | "
            f"Solver: {solver_times[i]:.6f}"
        )

        # Print to terminal
        print(line)

        # Save to file
        f.write(line + "\n")

print(f"\nResults saved to: {output_path}")