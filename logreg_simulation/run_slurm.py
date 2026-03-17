"""Slurm commands with relevant parameters to run simulations."""

alphas = [0.5, 0.9, 0.99]
pslabs = [0.1, 0.01, 0.001]

samplers = ["laplace", "gaussian", "zz"]

sampler_dict = {
    "laplace": {"step_size": {0.1: 0.02, 0.01: 0.05, 0.001: 0.1},
                "num_steps": {0.1: 125, 0.01: 50, 0.001: 25}},
    "gaussian": {"step_size": {0.1: 0.02, 0.01: 0.04, 0.001: 0.07},
                 "num_steps": {0.1: 50, 0.01: 25, 0.001: 14}},
    "zz": {"step_size": {0.5: 0.04, 0.9: 0.02, 0.99: 0.01},
           "num_steps": {0.5: 25, 0.9: 50, 0.99: 100}},
}

node_dict = {
    0.5: 165,
    0.9: 166,
    0.99: 167,
}
thin_dict = {
    0.1: 5,
    0.01: 50,
    0.001: 250,
}

for sampler in samplers:
    for pslab in pslabs:
        for alpha in alphas:
            iters = 15000 * thin_dict[pslab]
            thin = thin_dict[pslab]
            if sampler == "laplace":
                iters = iters // 5
                thin = thin // 5
            if pslab == 0.001 and alpha == 0.9:
                thin = thin // 5
            if sampler == "zz":
                print(
                    f"sbatch --nodelist=compute-{node_dict[alpha]} --array=0,1,2,3,4  --export=method={sampler},iters={iters},thin={thin},stepsize={sampler_dict[sampler]['step_size'][alpha]},steps={sampler_dict[sampler]['num_steps'][alpha]},pslab={pslab},alpha={alpha} --job-name={sampler}_{pslab}_{alpha} run_logreg_simulation.sh")
            else:
                print(
                    f"sbatch --nodelist=compute-{node_dict[alpha]} --array=0,1,2,3,4  --export=method={sampler},iters={iters},thin={thin},stepsize={sampler_dict[sampler]['step_size'][pslab]},steps={sampler_dict[sampler]['num_steps'][pslab]},pslab={pslab},alpha={alpha} --job-name={sampler}_{pslab}_{alpha} run_logreg_simulation.sh")
