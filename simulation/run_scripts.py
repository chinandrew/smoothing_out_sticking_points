import subprocess

n_iter = 25000
n = 2000
p = 2000
nonzero = 20
num_blocks = 20
sigma2 = 100
tau2 = 1

alphas = [0.5, 0.9, 0.99]

seeds = [0,1,2,3,4]
samplers = [
    ("shzz", 0.001, (2, 6), 50),
    ("szz", 0.001, 200, 1),
    ("szz-constant", 0.001, 200,1),
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
            command = ["sbatch --nodelist=compute-164 "] #  --mem=5G"]
            exports = [
                f"n_iter={n_iter}",
                f"n={n}",
                f"p={p}",
                f"nonzero={nonzero}",
                f"num_blocks={num_blocks}",
                f"tau2={tau2}",
                f"sigma2={sigma2}",
                f"thin={thin}",
            ]
            exports.append(f"method={method}")
            exports.append(f"seed={seed}")
            exports.append(f"alpha={alpha}")
            exports.append(f"p_slab={p_slab}")
            if method != "shzz":
                exports.append(f"t1={t}")
            else:
                exports.append(f"t1={t[0]}")
                exports.append(f"t2={t[1]}")
            job_name = f"{method}_{alpha}_{p_slab}_{seed}"
            command.append("--export=" + ",".join(exports))
            command.append(f"--job-name={job_name}")
            command.append(f"--time=7-00:00:00")
            command.append("run_simulation.sh")
            print(" ".join(command))
