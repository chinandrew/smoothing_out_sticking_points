"""Slurm commands with relevant parameters to run simulations."""

n_iter = 25000

alphas = [0.5, 0.9, 0.99]

seeds = [0, 1, 2, 3, 4]
samplers = [
    ("shzz", 0.001, (2, 6), 50),
    ("szz", 0.001, 200, 1),
    ("szz-constant", 0.001, 200, 1),
    ("shzz", 0.01, (2, 6), 5),
    ("szz", 0.01, 20, 1),
    ("szz-constant", 0.01, 20, 1),
    ("shzz", 0.1, (2, 6), 1),
    ("szz", 0.1, 4, 1),
    ("szz-constant", 0.1, 4, 1),
]

for alpha in alphas:
    for seed in seeds:
        for method, p_slab, t, thin in samplers:
            command = ["python linreg_simulation/run_simulation.py",]
            command.append(method)
            command.append(str(n_iter))
            command.append(str(thin))
            command.append(str(seed))
            command.append(str(p_slab))
            command.append(str(alpha))
            if method != "shzz":
                command.append(str(t))
            else:
                command.append(str(t[0]))
                command.append(str(t[1]))
            print(" ".join(command))
