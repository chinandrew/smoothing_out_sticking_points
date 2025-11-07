# Code accompanying the paper "Smoothing Out Sticking Points: Sampling from Discrete-Continuous Mixtures with Dynamical Monte Carlo by Mapping Discrete Mass into a Latent Universe" by Chin and Nishimura

Simulations used in the paper can be run through the `run_simulation.py` file in the `simulations/` folder, e.g. via
```
python run_simulation.py szz-constant 25 4 2000 2000 20 0.99 100 1 20 0.1 1 2
```
(see `run_simulation.py` for description of arguments).

`run_scripts.py` prints the commands to execute every job individually in SLURM via `run_simulations.sh`; an array job can be used if running more generally.

Results are store in the form of a pickled file containing the tuple
```
(samples, bounces, z_true, beta_true, runtime)
```
-`samples` - `n_iter x p` matrix of samples for the `p` regression coefficients.
-`bounces` - `n_iter x 2` matrix where first row is number of bounces for each iteration and second row is number of sticking/unsticking events.
-`z_true` - 0/1 array of true nonzero parameters.
-`beta_true` - array of true latent $\beta$ coefficients; `z_true * beta_true` is what gets multiplied by the design matrix to genreate data.
