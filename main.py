import argparse
import logging
from pathlib import Path
from corberan_data import CorberanData, CorberanAlgoType
import toy_instances as toy
import os
from CG_w_in_master import CG_w_in_master
from compact_model import CompactModel
import json
from corberan_solution import CorberanSolution

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='multi-time facility location allocation',
        description='command line interface for the multi-time facility location allocation')

    parser.add_argument("-b",
                        "--basedir",
                        dest="basedir", type=str, help="str: Base directory for input",
                        default ="",
                        required=False)

    parser.add_argument("-g",
                        "--epgap",
                        dest="epgap",
                        type=float,
                        help="float: LR gap threshold to stop CG loop. default to 1e-5",
                        default =1e-5,
                        required=False)

    parser.add_argument("-r",
                        "--redcostlimit",
                        dest="redcostlimit",
                        type=float,
                        help="float: threshold to consider a reduced cost to be negative. default to 1e-5",
                        default =1e-5,
                        required=False)

    parser.add_argument("-e",
                        "--example",
                        dest="runexample",
                        type=str,
                        help="str: run one of the toy examples: input 1, 2 or amelia",
                        default="",
                        required=False)

    parser.add_argument("-o",
                        "--outdir",
                        dest="outdir",
                        type=str,
                        help="str: Output directory for logging",
                        required=True)

    parser.add_argument("-f",
                        "--prefix",
                        dest="fileprefix",
                        type=str,
                        default="",
                        help="str: Input/output file prefix",
                        required=False)

    parser.add_argument("-v",
                        "--verbose",
                        dest="isverbose",
                        action="store_true",
                        help = "increase output verbosity",
                        required=False)

    parser.add_argument("-a",
                    "--algo",
                    dest="algolabel",
                    type=str,
                    help = "algorithm to run, expecting a label in set ['compact', 'cg_par', 'cg_seq']",
                    required=True)

    parser.add_argument("-z",
                        "--nonzerotolerance",
                        dest="nonzerotolerance",
                        type=float,
                        default=1e-5,
                        help="float: tolerance to consider a non-zero variable. default to 1e-5",
                        required=False)
    
    parser.add_argument("-t",
                        "--tilim",
                        dest="tilim",
                        type=int,
                        default=2000,
                        help="int: time limit (in seconds) of the compact model. default to 1200 seconds",
                        required=False)

    parser.add_argument("-m",
                        "--multipliers",
                        dest="mult_arg",
                        type=str,
                        help="str: multipliers for objective function components. either comma separated floats [r,f,a,m] or path to json file to read. default to 1 for all multipliers",
                        required=False)

    ##################### ADDED ####################
    parser.add_argument("-ga",
                        "--usensga",
                        dest="use_nsga",
                        action="store_true",
                        help="use NSGA-II genetic algorithm from CG columns",
                        required=False)
    
    parser.add_argument("-cl",
                        "--gamincol",
                        dest="nsga_min_cols_per_t",
                        type=int,
                        default=30,
                        help="min number of columns per time before stopping CG in GA mode",
                        required=False)
    parser.add_argument("-pop",
                        "--populationsize",
                        dest="nsga_pop_size",
                        type=int,
                        default=30,
                        help="initial population size for GA",
                        required=False)
    parser.add_argument("-gen",
                        "--numgenerations",
                        dest="nsga_n_gen",
                        type=int,
                        default=30,
                        help="number of generations for GA",
                        required=False)
    parser.add_argument("--seed",
                    dest="seed",
                    type=int,
                    default=0,
                    help="random seed for reproducibility",
                    required=False)
    parser.add_argument("--poolfrac",
                    dest="nsga_pool_frac",
                    type=float,
                    default=1.0,
                    help="CG percentage in initial pool for GA",
                    required=False)
    parser.add_argument("--cgpool",
                        dest="do_cg_pool",
                        action="store_true",
                        help="get a number of columns from CG to store",
                        required=False)
    parser.add_argument("--poolfile",
                        dest="pool_file", 
                        type=str,
                        default="", 
                        help="Path for pool .npz saved after CG")
    parser.add_argument("--heurpoolfile",
                        dest="heur_pool_file", 
                        type=str,
                        default="", 
                        help="Path for heur pool .npz saved after CG")
    parser.add_argument("--colsfile",
                        dest="cols_file", 
                        type=str,
                        default="", 
                        help="Path for cols .pkl saved after CG")

    ################################################
    
    _cmd_options = parser.parse_args()
    # BASEDIR = "/home/marco/Documents/work/corberan/logs/"

    # -----------------------------
    # Logging setup
    # -----------------------------

    algotype = CorberanAlgoType.from_label(_cmd_options.algolabel)

    RUN_DIR = Path(_cmd_options.outdir)
    RUN_DIR.mkdir(exist_ok=True, parents=True)

    # read multipliers
    multipliers = {
        "cf": 1,
        "cr": 1,
        "ca": 1,
        "cm": 1
    }

    if _cmd_options.mult_arg is not None:
        if _cmd_options.mult_arg.endswith('json'):
            with open(os.path.join(_cmd_options.basedir, _cmd_options.mult_arg), 'r') as fin:
                j = json.load(fin)
                multipliers = {
                    "cf": j.get("open", 1),
                    "cr": j.get("run", 1),
                    "ca": j.get("assign", 1),
                    "cm": j.get("relocate", 1)
                }
        else:
            mult_values = [float(el) for el in _cmd_options.mult_arg.split(",")]
            multipliers = {
                "cr": mult_values[0],
                "cf": mult_values[1],
                "ca": mult_values[2],
                "cm": mult_values[3]
            }

    mult_label = ""
    for il, l in enumerate(["r", "f", "a", "m"]):
        mult_label += l + "%d" % (multipliers[f"c{l}"])
        if il < len(multipliers) - 1:
            mult_label += "_"

    logfileprefix = "run"
    if  _cmd_options.fileprefix != "":
        logfileprefix = _cmd_options.fileprefix

    logfileprefix = logfileprefix + "_" + _cmd_options.algolabel + "_" + mult_label

    logging.basicConfig(
        filename=f"{_cmd_options.outdir}{logfileprefix}.log",
        filemode="w",
        format="{asctime}.s{msecs:.0f} - {levelname} - {message}",
        style="{",
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    # READ DATA
    # self.cData = set_p.example2()
    _cData = None

    if _cmd_options.runexample == "1":
        _cData = toy.example1()
    elif _cmd_options.runexample == "2":
        _cData = toy.example2()
    elif _cmd_options.runexample == "amelia":
        _cData = toy.dataset_amelia()

    if _cData is None:
        if _cmd_options.basedir != "" and _cmd_options.fileprefix != "" and os.path.exists(_cmd_options.basedir):

            _cData = CorberanData.from_json_params(
                _cmd_options.basedir,
                _cmd_options.fileprefix)
            
    if _cData is not None:

        logging.info(f"running {_cmd_options.algolabel}")
        logging.info("data read with success")

        _cData.multipliers = multipliers
        _cData.set_log_out_dir(RUN_DIR)
        _cData.debug_info = _cmd_options.isverbose
        _cData.nonzero_tolerance = _cmd_options.nonzerotolerance

        ##################### ADDED ####################
        _cData.use_nsga = _cmd_options.use_nsga
        _cData.nsga_min_cols_per_t = _cmd_options.nsga_min_cols_per_t
        _cData.nsga_pop_size = _cmd_options.nsga_pop_size
        _cData.nsga_n_gen = _cmd_options.nsga_n_gen
        _cData.seed = _cmd_options.seed
        _cData.nsga_pool_frac = _cmd_options.nsga_pool_frac
        _cData.do_cg_pool = _cmd_options.do_cg_pool
        _cData.pool_file = _cmd_options.pool_file
        _cData.heur_pool_file = _cmd_options.heur_pool_file
        _cData.cols_file = _cmd_options.cols_file


        ################################################

        sol = None

        _cData.outfilename  = logfileprefix

        solfilename = os.path.join(_cmd_options.outdir, _cData.outfilename) +  "_sol.json"
        if algotype in [CorberanAlgoType.CG_SEQ, CorberanAlgoType.CG_PAR]:
            if algotype == CorberanAlgoType.CG_PAR:
                cg_algo = CG_w_in_master(_cData, True)
            else:
                cg_algo = CG_w_in_master(_cData, False)

            sol = cg_algo.cg_loop(
                _cmd_options.epgap, 
                _cmd_options.redcostlimit,
                _cmd_options.tilim
                )
                
        elif algotype in [CorberanAlgoType.COMPACT, CorberanAlgoType.ROOT, CorberanAlgoType.LP_REL]:
            compact = CompactModel(_cData)

            _root = algotype == CorberanAlgoType.ROOT
            lp_rel = False
            if not _root:
                lp_rel = algotype == CorberanAlgoType.LP_REL

            sol = compact.run(_cmd_options.tilim,
                              _root,
                              lp_rel)

        elif algotype in [CorberanAlgoType.GA_ONLY]:
            if _cmd_options.pool_file == "":
                raise ValueError("GA_ONLY richiede --poolfile <path_al_pool.npz>")
            if _cmd_options.heur_pool_file == "":
                raise ValueError("GA_ONLY richiede --heurpoolfile <path_al_pool.npz>")

            import helper as hlp
            pool_loaded = hlp.load_population(_cmd_options.pool_file)
            heur_pool_loaded = hlp.load_population(_cmd_options.heur_pool_file)
            cols_loaded = hlp.load_columns(_cmd_options.cols_file)
            cg_algo = CG_w_in_master(_cData, False)

            sol = cg_algo.ga_only(
                cols_loaded,
                pool_loaded,
                heur_pool_loaded,
                frac=_cmd_options.nsga_pool_frac,
                seed=_cmd_options.seed,
            )

        elif algotype in [CorberanAlgoType.HEUR_ONLY]:
            heur_sol = CorberanSolution.construction_heuristic(_cData)

            out_txt = Path(_cmd_options.outdir) / "heur_results.txt"

            with open(out_txt, "a") as f:
                if not heur_sol.isfeasible:
                    f.write(f"{_cmd_options.fileprefix};INFEASIBLE\n")
                else:
                    f.write(f"{_cmd_options.fileprefix};{heur_sol.objval}\n")
                
        elif algotype in [CorberanAlgoType.DRMP_ONLY]:
            if _cmd_options.cols_file == "":
                raise ValueError("DRMP_ONLY richiede --colsfile <path_al_cols.npz>")
            
            import helper as hlp
            cols_loaded = hlp.load_columns(_cmd_options.cols_file)
            cg_algo = CG_w_in_master(_cData, False)

            sol = cg_algo.drmp_only(
                cols_loaded
            )

    else:

        logging.error("no input chosen. exit execution.")
