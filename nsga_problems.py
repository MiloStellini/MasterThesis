import numpy as np
from corberan_data import CorberanData, PricerTimeColumn
from typing import List
from nsga_config import NsgaProblemConfig, parse_index_q, parse_index_x, parse_index_y, split_chromosome, x_build_from_yq

########## PROBLEMA SOLO Y ##########

def facility_opening_fitness(chrom: np.ndarray, cData: CorberanData, t: int) -> float:
    y = chrom.astype(bool)
    cost = 0.0

    for f in range(cData.nF):
        cost += cData.get_mult_cr() * cData.get_cr(f, t-1) * y[f]

    if y.sum() == 0:
        cost += 1e6

    return cost


def column_to_chromosome_y(col: PricerTimeColumn, cData: CorberanData) -> np.ndarray:
    chrom = np.zeros(cData.nF, dtype=bool)

    for name, val in col.nonzero_vars.items():
        if val <= 0.5:
            continue
        f = parse_index_y(name)
        if f is not None:
            chrom[f] = True

    return chrom


def header_vars_y(cData: CorberanData) -> List[str]:
    return [f"y_{f}" for f in range(cData.nF)]



########## PROBLEMA Y + X ##########

def facility_opening_with_x_fitness(chrom:np.ndarray, cData: CorberanData, t: int) -> float:
    nF = cData.nF
    nC = cData.nC

    y = chrom[:nF].astype(bool)
    x = chrom[nF:].reshape(nF,nC)

    cost = 0.0

    for f in range(nF):
        cost += cData.get_mult_cr() * cData.get_cr(f, t-1) * y[f]        
        for c in range(nC):
            cost += cData.get_mult_ca() * cData.ca[f, c] * x[f, c]

    cost += (y.sum() == 0) * 1e5
    for c in range(nC):
        cost += (sum(x[f, c] for f in range(nF)) < 1)
    return cost

    
def column_to_chromosome_yx(col: PricerTimeColumn, cData: CorberanData) -> np.ndarray:
    nF = cData.nF
    nC = cData.nC
    chrom = np.zeros(nF + nF * nC, dtype=bool)

    for name, val in col.nonzero_vars.items():
        if val <= 0.5:
            continue

        f = parse_index_y(name)
        if f is not None:
            chrom[f] = True
            continue

        ij = parse_index_x(name)
        if ij is not None:
            i, j = ij
            idx = nF + i * nC + j
            chrom[idx] = True
            continue

    return chrom   


def header_vars_yx(cData: CorberanData) -> List[str]:
    nF = cData.nF
    nC = cData.nC

    header = []
    header.extend([f"y_{f}" for f in range(nF)])
    header.extend([f"x_{i}_{j}" for i in range(nF) for j in range(nC)])
    return header



########## PROBLEMA CG (X + Y + Q) ##########
def column_to_chromosome_blocks(col: PricerTimeColumn, cData: CorberanData) -> np.ndarray:
    nF = cData.nF
    nC = cData.nC
   
    chrom = np.zeros(nF + nF + (nF * nC))

    for name, val in col.nonzero_vars.items():
        if val == 0:
            continue
        
        f = parse_index_y(name)
        if f is not None:
            chrom[f] = val
            continue

        f = parse_index_q(name)
        if f is not None:
            chrom[nF + f] = val
            continue
        
        ij = parse_index_x(name)
        if ij is not None:
            i, j = ij
            chrom[nF + nF + i * nC + j] = val
            continue
    return chrom


def fitness_blocks(chrom:np.ndarray, cData: CorberanData, t: int) -> float:
    nF , nC = cData.nF, cData.nC

    d = cData.d[:, t]
    ca = cData.ca
    cr = cData._cr

    y, q, x = split_chromosome(chrom, cData)

    cost_open = cData.get_mult_cr() * np.dot(cr, y)
    cost_assign = cData.get_mult_ca() * np.sum(ca * x)
    cost = cost_open + cost_assign

    penalty = 0.0

    penalty += (y.sum() == 0) * 1e6

    excess_q = q.sum() - cData.M
    penalty += (excess_q > 0.0) * (excess_q / cData.M) * 1e5

    deficit = 1.0 - x.sum(axis=0)
    mask_cov = (deficit * d > 0.0)
    penalty += np.sum(deficit[mask_cov] * 1e5)

    load = x @ d
    overflow = load -q
    overflow_pos = overflow[overflow > 0.0]
    if overflow_pos.size > 0:
        penalty += np.sum(overflow_pos * 1e5)
    
    return cost + penalty

def header_vars_blocks(cData: CorberanData) -> List[str]:
    nF = cData.nF
    nC = cData.nC

    header = []
    header.extend([f"y_{f}" for f in range(nF)])
    header.extend([f"q_{f}" for f in range(nF)])
    header.extend([f"x_{i}_{j}" for i in range(nF) for j in range(nC)])
    return header


########## PROBLEMA CG (Y + Q) ##########

def column_to_chromosome_YQ(col: PricerTimeColumn, cData: CorberanData) -> np.ndarray:
    nF = cData.nF
   
    chrom = np.zeros(nF + nF, dtype=float)

    for name, val in col.nonzero_vars.items():
        if val == 0:
            continue
        
        f = parse_index_y(name)
        if f is not None:
            chrom[f] = val
            continue

        f = parse_index_q(name)
        if f is not None:
            chrom[nF + f] = val
            continue
    return chrom

def fitness_YQ(chrom:np.ndarray, cData: CorberanData, t: int) -> float:

    nF = cData.nF

    y = chrom[:nF]
    q = chrom[nF:]
    x, feasible = x_build_from_yq(y, q, cData, t)
    if not feasible:
        return 1e6
    
    cost = cData.get_mult_cr() * np.dot(cData._cr, (y > 0.5).astype(float))
    cost += cData.get_mult_ca() * np.sum(cData.ca * x)

    penalty = (y.sum() == 0) * 1e5
    # excess_q = q.sum() - cData.M
    # penalty += (excess_q > 0.0) * (excess_q / cData.M) * 1e4
    
    return cost + penalty


def header_vars_YQ(cData: CorberanData) -> List[str]:
    nF = cData.nF

    header = []
    header.extend([f"y_{f}" for f in range(nF)])
    header.extend([f"q_{f}" for f in range(nF)])
    return header


facility_opening_config = NsgaProblemConfig(
    name="y_only",
    get_n_var=lambda cData: cData.nF,
    column_to_chromosome=column_to_chromosome_y,
    fitness=facility_opening_fitness,
    get_header_vars=header_vars_y
)

facility_opening_yx_config = NsgaProblemConfig(
    name="y_x",
    get_n_var=lambda cData: cData.nF + cData.nF * cData.nC,
    column_to_chromosome=column_to_chromosome_yx,
    fitness=facility_opening_with_x_fitness,
    get_header_vars=header_vars_yx,
)

facility_opening_mix_config = NsgaProblemConfig(
    name="mix",
    get_n_var=lambda cData: cData.nF + cData.nF + cData.nF * cData.nC,
    column_to_chromosome=column_to_chromosome_blocks,
    fitness=fitness_blocks,
    get_header_vars=header_vars_blocks,
)

facility_opening_yq_config = NsgaProblemConfig(
    name="y_q",
    get_n_var=lambda cData: cData.nF + cData.nF,
    column_to_chromosome=column_to_chromosome_YQ,
    fitness=fitness_YQ,
    get_header_vars=header_vars_YQ,
)

# def debug_fitness_for_column(col: PricerTimeColumn, cData: CorberanData, t: int):
#     chrom = column_to_chromosome_blocks(col, cData)
#     y, q, x = split_chromosome(chrom, cData)

#     nF, nC = cData.nF, cData.nC
#     d = cData.d

#     # Ricalcolo separato dei pezzi
#     cost = 0.0
#     for f in range(nF):
#         cost += cData.get_mult_cr() * cData.get_cr(f, t-1) * y[f]
#         for c in range(nC):
#             cost += cData.get_mult_ca() * cData.ca[f, c] * x[f, c]

#     pen_y0 = (y.sum() == 0) * 1e6

#     excess_q = float(q.sum() - cData.M)
#     pen_M = (excess_q > 0.0) * (excess_q / cData.M) * 1e5

#     pen_cov = 0.0
#     for c in range(nC):
#         deficit = 1 - float(np.sum(x[:, c]))
#         pen_cov += (deficit * d[c, t] > 0.0) * deficit * 1e5

#     pen_cap = 0.0
#     for f in range(nF):
#         overflow = float(sum(d[c, t] * x[f, c] for c in range(nC))) - q[f]
#         pen_cap += (overflow > 0.0) * overflow * 1e5

#     total = cost + pen_y0 + pen_M + pen_cov + pen_cap

#     print(f"col = {col.col_name}, t={t}")
#     print(f"  cost    = {cost}")
#     print(f"  pen_y0  = {pen_y0}")
#     print(f"  pen_M   = {pen_M}")
#     print(f"  pen_cov = {pen_cov}")
#     print(f"  pen_cap = {pen_cap}")
#     print(f"  total   = {total}")