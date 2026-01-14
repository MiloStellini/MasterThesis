import numpy as np
from typing import Dict, Optional
from corberan_data import CorberanData

from pymoo.algorithms.moo.nsga2 import NSGA2
#from pymoo.operators.crossover.pntx import TwoPointCrossover
#from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover
from pymoo.termination import get_termination
from pymoo.core.callback import Callback

import csv
from pathlib import Path
from nsga_config import NsgaProblemConfig, split_chromosome, join_chromosome

def _create_YQ_initial_population(cData, n_extra: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    nF = cData.nF
    M = cData.M

    y = (rng.random((n_extra, nF)) < 0.5).astype(float)
    zero_rows = (y.sum(axis=1) == 0)
    if np.any(zero_rows):
        idx = rng.integers(0, nF, size=int(zero_rows.sum()))
        y[zero_rows, idx] = 1.0

    q = rng.random((n_extra, nF)) * M

    return np.concatenate([y, q], axis=1)

class LogBestFCallback(Callback):
    def __init__(self, t: int):
        super().__init__()
        self.t = t
        self.history = []

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F").flatten().astype(float)
        best_f = float(F.min())
        mean_f = float(F.mean())
        self.history.append((self.t, gen, best_f, mean_f))

class InitialSampling(Sampling):
    def __init__(self, init_pop: np.ndarray, config: NsgaProblemConfig):
        super().__init__()
        self.init_pop = init_pop
        self.config = config

    def _do(self, problem, n_samples, **kwargs):

        n_var = problem.n_var
        pop = np.zeros((n_samples, n_var), dtype=float)

        k = min(len(self.init_pop), n_samples)
        pop[:k, :] = self.init_pop[:k, :]

        if k < n_samples:
            n_extra = n_samples - k
            if self.config.name not in ["y_q","mix"]:
                extra = np.random.randint(0, 2, size=(n_samples - k, n_var))

                for i in range(extra.shape[0]):
                    if extra[i].sum() == 0:
                        idx = np.random.randint(0, n_var)
                        extra[i, idx] = 1
                
            else:
                extra = _create_YQ_initial_population(problem.cData, n_extra)

            pop[k:, :] = extra    
        return pop
    
    def _create_mix_initial_population(self, problem, n_extra: int) -> np.ndarray:

        cData = problem.cData
        nF, nC = cData.nF, cData.nC
        M = cData.M

        y = (np.random.rand(n_extra, nF) < 0.5).astype(float)
        zero_rows = (y.sum(axis=1) == 0)
        if np.any(zero_rows):
            idx = np.random.randint(0, nF, size=zero_rows.sum())
            y[zero_rows, idx] = 1.0

        q = np.random.rand(n_extra, nF) * M

        x = np.random.rand(n_extra, nF * nC)
        flip = np.repeat(~y.astype(bool), nC, axis=1)
        x[flip] = 0.0

        return np.concatenate([y, q, x], axis=1)
    
class FacilityOpeningProblem(ElementwiseProblem):
    def __init__(self, cData: CorberanData, t: int, config:NsgaProblemConfig):
        super().__init__(
            n_var=config.get_n_var(cData),
            n_obj=1,
            n_constr=0,
            xl=0,
            xu=1
        )
        self.cData = cData
        self.t = t
        self.config = config

    def _evaluate(self, x, out, *args, **kwargs):
        chrom = np.array(x)
        cost = self.config.fitness(chrom, self.cData, self.t)
        out["F"] = cost


class MixedMutation(Mutation):
    def __init__(self, p_y, p_q, p_x):
        super().__init__()
        self.p_y = p_y
        self.p_q = p_q
        self.p_x = p_x

    def _do(self, problem, pop, **kwargs):
        pop = pop.copy()
        n_pop, _ = pop.shape

        cData = problem.cData
        nF, nC = cData.nF, cData.nC
        M = cData.M

        y = pop[:, :nF]
        q = pop[:, nF:2*nF]
        x = pop[:, 2*nF:].reshape(n_pop, nF, nC)

        y_mut = (y > 0.5).astype(int)
        q_mut = q.copy()
        x_mut = x.copy()

        # Mutazione y
        flip_y = np.random.rand(n_pop, nF) < self.p_y
        y_mut[flip_y] = 1 - y_mut[flip_y]

        zero_rows = (y_mut.sum(axis=1) == 0)
        if np.any(zero_rows):
                idx = np.random.randint(0, nF, size=zero_rows.sum())
                y_mut[zero_rows, idx] = 1

        opened = (y_mut == 1)
        x_mut *= opened[:, :, None]

        # # Serve davvero questo blocco???
        # newly_open = (y_mut == 1) & (y <= 0.5)
        # inds, fs = np.where(newly_open)
        # for ind, f in zip(inds, fs):
        #     x_mut[ind, f, :] = np.random.rand(nC) * 0.2
        #     if q_mut[ind, f] == 0:
        #         q_mut[ind, f] = np.random.rand() * M * 0.1

        # Mutazione q
        flip_q = np.random.rand(n_pop, nF) < self.p_q
        noise_q = (np.random.rand(n_pop, nF) - 0.5) * (M * 0.2)
        q_mut[flip_q] += noise_q[flip_q]
        q_mut = np.clip(q_mut, 0.0, M)

        # Mutazione x
        flip_x = np.random.rand(n_pop, nF, nC) < self.p_x
        flip_x &= (y_mut == 1)[:, :, None]
        noise_x = (np.random.rand(n_pop, nF, nC) - 0.5) * 0.2
        x_mut[flip_x] += noise_x[flip_x]
        x_mut = np.clip(x_mut, 0.0, 1.0)

        pop[:, :nF] = y_mut
        pop[:, nF:2*nF] = q_mut
        pop[:, 2*nF:] = x_mut.reshape(n_pop, nF * nC)

        return pop
    
class YQMutation(Mutation):
    def __init__(self, p_y, p_q):
        super().__init__()
        self.p_y = p_y
        self.p_q = p_q

    def _do(self, problem, pop, **kwargs):
        pop = pop.copy()
        n_pop, _ = pop.shape

        cData = problem.cData
        nF = cData.nF
        M = cData.M

        y = pop[:, :nF]
        q = pop[:, nF:2*nF]

        y_mut = (y > 0.5).astype(int)

        # Mutazione y
        flip_y = np.random.rand(n_pop, nF) < self.p_y
        y_mut[flip_y] = 1 - y_mut[flip_y]

        zero_rows = (y_mut.sum(axis=1) == 0)
        if np.any(zero_rows):
                idx = np.random.randint(0, nF, size=zero_rows.sum())
                y_mut[zero_rows, idx] = 1

        # Mutazione q         
        # flip_q = np.random.rand(n_pop, nF) < self.p_q
        # noise_q = (np.random.rand(n_pop, nF) - 0.5) * (M * 0.05)
        # q[flip_q] += noise_q[flip_q]
        # q = np.clip(q, 0.0, M)
        mask_q = (np.random.rand(n_pop, nF) < self.p_q) & (q > 0)
        I, F = np.where(mask_q)
        if I.size > 0:
            R = np.random.randint(0, nF-1, size=I.size)
            G = R + (R >= F)

            delta = np.random.rand(I.size) * q[I, F]

            q[I, F] -= delta
            np.add.at(q, (I, G), delta)

        pop[:, :nF] = y_mut
        pop[:, nF:] = q
        return pop
            

class MixedCrossover(Crossover):
    def __init__(self, prob):
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # X.shape = (n_parents, n_matings, n_var)
        
        cData = problem.cData
        nF = cData.nF
        nC = cData.nC

        n_var = X.shape[2]
        n_matings = X.shape[1]

        off = np.empty((self.n_offsprings, n_matings, n_var))

        for k in range(n_matings):
            p1 = X[0, k, :].copy()
            p2 = X[1, k, :].copy()

            c1 = p1.copy()
            c2 = p2.copy()

            if np.random.rand() < self.prob:

                swap = np.random.rand(nF) < 0.5

                for f in range(nF):
                    if not swap[f]:
                        continue

                    y_idx = f
                    q_idx = nF + f
                    x_start = nF + nF + f * nC
                    x_end = x_start + nC

                    c1[y_idx], c2[y_idx] = c2[y_idx], c1[y_idx]

                    c1[q_idx], c2[q_idx] = c2[q_idx], c1[q_idx]
                    
                    aux = c1[x_start:x_end].copy()
                    c1[x_start:x_end] = c2[x_start:x_end]
                    c2[x_start:x_end] = aux

                
            off[0, k, :] = c1
            off[1, k, :] = c2

        return off
    

class YQCrossover(Crossover):
    def __init__(self, prob):
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # X.shape = (n_parents, n_matings, n_var)
        
        nF = problem.cData.nF

        n_var = X.shape[2]
        n_matings = X.shape[1]

        off = np.empty((self.n_offsprings, n_matings, n_var))

        for k in range(n_matings):
            p1 = X[0, k, :].copy()
            p2 = X[1, k, :].copy()

            c1 = p1.copy()
            c2 = p2.copy()

            if np.random.rand() < self.prob:

                swap = np.random.rand(nF) < 0.5
                for f in np.where(swap)[0]:
                    c1[f], c2[f] = c2[f], c1[f]
                    c1[nF+f], c2[nF+f] = c2[nF+f], c1[nF+f]
                
            off[0, k, :] = c1
            off[1, k, :] = c2

        return off


def extract_population(all_columns: Dict[int, list],
                       cData: CorberanData,
                       config: NsgaProblemConfig,
                       drop_duplicates: bool = False) -> Dict[int, list]:
    
    population = {t: [] for t in cData.T}

    for t, cols in all_columns.items():
        for col in cols:
            chrom = config.column_to_chromosome(col, cData)
            population[t].append(chrom)

    if drop_duplicates:
        for t in population:
            uniq = []
            seen = set()
            for chrom in population[t]:
                key = tuple(chrom.tolist())
                if key not in seen:
                    seen.add(key)
                    uniq.append(chrom)
            population[t] = uniq

    return population




def run_nsga(population: Dict[int, list],
             cData: CorberanData,
             config: NsgaProblemConfig):
    
    base_seed = getattr(cData, "seed", None)
    if base_seed is not None:
        np.random.seed(base_seed)
    
    results = {}
    generations = {}

    for t, t_cols in population.items():
        if not t_cols:
            continue

        if base_seed is not None:
            seed_t = base_seed + 1000 * int(t) + 7
            np.random.seed(seed_t)
        
        init_pop = np.vstack(t_cols)

        problem = FacilityOpeningProblem(cData, t, config)

        sampling = InitialSampling(init_pop=init_pop, config=config)

        algorithm = NSGA2(pop_size=cData.nsga_pop_size,
                          sampling=sampling,
                          crossover=YQCrossover(prob=0.9),
                          mutation=YQMutation(p_y=0.1, p_q=0.1),
                          eliminate_duplicates=True)

        termination = get_termination("n_gen", cData.nsga_n_gen)
        cb = LogBestFCallback(t)
        res = minimize(problem,
                       algorithm,
                       termination,
                       callback=cb,
                       verbose=False,
                       seed=(seed_t if base_seed is not None else None))
        _log_nsga_generation_history(t, cb.history, cData, config)
        generations[t] = res.algorithm.n_gen

        X = res.pop.get("X")
        F = res.pop.get("F").flatten()

        if X.ndim == 1:
            X = X.reshape(1, -1)
        _log_nsga_result_for_t(t, X, F, cData, config)

        results[t] = X

    return results, generations


def _log_nsga_result_for_t(t: int,
                           pop: np.ndarray,
                           fitness: np.ndarray,
                           cData: CorberanData,
                           config: NsgaProblemConfig):
    
    pop = pop.astype(float)
    outdir = Path(cData.log_out_dir)
    outpath = outdir / f"{cData.outfilename}_nsga_{config.name}_{t}.csv"

    outdir.mkdir(parents=True, exist_ok=True)
    
    header = ["t", "id", "cost"] + config.get_header_vars(cData)

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(pop.shape[0]):
            row = [t, i, float(fitness[i])] + pop[i].tolist()
            writer.writerow(row)

def _log_nsga_generation_history(t: int,
                                 history,
                                 cData: CorberanData,
                                 config: NsgaProblemConfig):
    outdir = Path(cData.log_out_dir)
    outpath = outdir / f"{cData.outfilename}_nsga_gen_{config.name}_{t}.csv"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "gen", "best_f", "mean_f"])
        for row in history:

            writer.writerow(row)    
