import subprocess
import json
import os
import sys
from pathlib import Path
import time
import numpy as np
import logging

# from corberan_data import CorberanData
import toy_instances as toy

EXECUTABLE_PATH = Path(__file__).parent / "construction_heuristic"


def construction_heuristic(nF, nC, nT, d, q_in):

    # Variables
    x = np.zeros((nF, nC, nT))
    y = np.ones((nF, nT))
    w = np.zeros((nF, nF, nT))
    q = np.tile(q_in[:, None], (1, nT))
    z = np.zeros((nF, nT))

    tot_res = q_in.sum()
    # Check for feasibility
    for t in range(nT):
        if tot_res < sum(d[c, t] for c in range(nC)):
            logging.error(f"Not enough resources for demand at time {t}. Problem is infeasible.")
            exit()

    # TODO: parallelizza il loop per ogni t \in T eseguendo il codice attuale
    # oppure sempre parallelizzando il loop per ogni t \in T, ogni singolo T è un problema di flusso
    # aprendo ogni facility, che può essere risolto all'ottimo in tempo polinomiale
    # networkx
    for t in range(nT):
        for c in range(nC):
            already_assigned = 0

            f = 0
            residual_capacity = q_in[f] - sum(x[f, k, t] for k in range(nC))

            while already_assigned < d[c, t]:
                if residual_capacity > d[c, t] - already_assigned:
                    x[f, c, t] += d[c, t] - already_assigned
                    already_assigned = d[c, t]
                else:
                    if residual_capacity > 0:
                        x[f, c, t] += residual_capacity
                        already_assigned += residual_capacity
                    f += 1
                    if f < nF:
                        residual_capacity = q_in[f] - sum(x[f, k, t] for k in range(nC))

    for t in range(nT):
        for j in range(nC):
            if x[:, j, t].sum() != 0:
                for i in range(nF):
                    x[i, j, t] = x[i, j, t] / x[:, j, t].sum()

    return x, y, w, q, z

class HeuristicRunner:
    """Wrapper to run the C++ construction heuristic and handle results."""
    
    def __init__(self, executable_path):
        """
        Initialize the runner.
        
        Args:
            executable_path: Path to the compiled C++ executable
        """
        self.executable_path = Path(executable_path)
        
        if not self.executable_path.exists():
            raise FileNotFoundError(f"Executable not found: {executable_path}")
        
        if not os.access(self.executable_path, os.X_OK):
            raise PermissionError(f"Executable is not executable: {executable_path}")
    
    def create_input_file(self, nF, nC, nT, d, q_in, input_file):
        """
        Create input JSON file for the C++ program.
        
        Args:
            nF: Number of facilities
            nC: Number of customers
            nT: Number of time periods
            d: Demand matrix (nC x nT)
            q_in: Initial capacity vector (nF)
            input_file: Path where to save the input JSON
        """
        input_data = {
            "nF": nF,
            "nC": nC,
            "nT": nT,
            "d": d.tolist(),
            "q_in": q_in.tolist()
        }
        
        with open(input_file, 'w') as f:
            json.dump(input_data, f, indent=2)
        
        print(f"Input file created: {input_file}")
    
    def _run(self, input_file, output_file, num_threads=None, timeout=None) -> dict:
        """
        Run the C++ heuristic executable.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path where output JSON will be written
            num_threads: Number of OpenMP threads (optional)
            timeout: Maximum execution time in seconds (optional)
            
        Returns:
            dict: Result containing status and execution info
        """
        # Build command
        cmd = [
            str(self.executable_path),
            '--input', str(input_file),
            '--output', str(output_file)
        ]
        
        if num_threads is not None:
            cmd.extend(['--threads', str(num_threads)])
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            
            # Run the C++ executable
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"Execution completed successfully in {elapsed_time:.2f} seconds")
            print(f"stdout:\n{result.stdout}")
            
            if result.stderr:
                print(f"stderr:\n{result.stderr}")
            
            return {
                'success': True,
                'elapsed_time': elapsed_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            print(f"Error: Execution timed out after {timeout} seconds")
            return {
                'success': False,
                'error': 'timeout',
                'timeout': timeout
            }
            
        except subprocess.CalledProcessError as e:
            print(f"Error: Execution failed with return code {e.returncode}")
            print(f"stdout:\n{e.stdout}")
            print(f"stderr:\n{e.stderr}")
            return {
                'success': False,
                'error': 'execution_failed',
                'return_code': e.returncode,
                'stdout': e.stdout,
                'stderr': e.stderr
            }
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return {
                'success': False,
                'error': 'unexpected',
                'message': str(e)
            }
    
    def _read_output(self, output_file : str) -> dict:
        """
        Read the output JSON file produced by the C++ program.
        
        Args:
            output_file: Path to the output JSON file
            
        Returns:
            dict: Dictionary containing x, y, w, q, z arrays
        """
        try:
            with open(output_file, 'r') as f:
                output_data = json.load(f)
            
            # print(f"Output file read successfully: {output_file}")
            
            # Convert to numpy arrays if numpy is available
            
            output_data = {
                'x': np.array(output_data['x']),
                'y': np.array(output_data['y']),
                'w': np.array(output_data['w']),
                'q': np.array(output_data['q']),
                'z': np.array(output_data['z'])
            }
            # print("Output converted to numpy arrays")
        
            return output_data
            
        except FileNotFoundError:
            logging.error(f"Error: Output file not found: {output_file}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error: Invalid JSON in output file: {e}")
            return None
        except Exception as e:
            logging.error(f"Error reading output: {str(e)}")
            return None
    
    def run_and_get_results(self, input_file, output_file, num_threads=None, timeout=None):
        """
        Convenience method: run the executable and return the results.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path where output JSON will be written
            num_threads: Number of OpenMP threads (optional)
            timeout: Maximum execution time in seconds (optional)
            
        Returns:
            tuple: (execution_info, results) or (execution_info, None) if failed
        """
        execution_info = self._run(input_file, output_file, num_threads, timeout)
        
        if execution_info['success']:
            results = self._read_output(output_file)
            return execution_info, results
        else:
            return execution_info, None

def construction_heuristic_cpp(nF, nC, nT, d, q_in):
    input_file = 'input.json'
    output_file = 'output.json'
    hr = HeuristicRunner(EXECUTABLE_PATH)
    hr.create_input_file(nF, nC, nT, d, q_in, input_file)
    _, hc = hr.run_and_get_results(input_file, output_file)

    return hc["x"], hc["y"], hc["w"], hc["q"], hc["z"]

def toy_example_cpp():
    """Example usage of the HeuristicRunner."""
    
    # Configuration
    executable_path = EXECUTABLE_PATH  # Path to your compiled C++ executable
    input_file = 'input.json'
    output_file = 'output.json'
    
    # Example problem data
    nF = 3
    nC = 4
    nT = 5
    d = [
        [10.0, 12.0, 15.0, 11.0, 13.0],
        [8.0, 9.0, 11.0, 10.0, 12.0],
        [15.0, 14.0, 16.0, 13.0, 15.0],
        [7.0, 8.0, 9.0, 8.0, 10.0]
    ]
    q_in = [50.0, 45.0, 40.0]
    
    try:
        # Create runner
        runner = HeuristicRunner(executable_path)
        
        # Create input file
        runner.create_input_file(nF, nC, nT, d, q_in, input_file)
        
        # Run and get results
        exec_info, results = runner.run_and_get_results(
            input_file=input_file,
            output_file=output_file,
            num_threads=4,
            timeout=60  # 60 seconds timeout
        )
        
        if results is not None:
            print("\n=== Results ===")
            print(f"x shape: {len(results['x'])} x {len(results['x'][0])} x {len(results['x'][0][0])}")
            print(f"y shape: {len(results['y'])} x {len(results['y'][0])}")
            print(f"Execution time: {exec_info['elapsed_time']:.2f} seconds")
            
            # Example: access specific values
            print(f"\nSample values from x[0][0]: {results['x'][0][0][:3]}")
        else:
            print("\nFailed to get results")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def main():    
    amelia_data = toy.dataset_amelia()

    start_t = time.time()
    x, y, w, q, z = construction_heuristic(amelia_data.nF, amelia_data.nC, amelia_data.nt, amelia_data.d, amelia_data.q_in)
    print(f"ending python implementation in {time.time() - start_t}")

    start_t = time.time()
    input_file = 'input.json'
    output_file = 'output.json'
    hr = HeuristicRunner(EXECUTABLE_PATH)
    hr.create_input_file(amelia_data.nF, amelia_data.nC, amelia_data.nt, amelia_data.d, amelia_data.q_in, input_file)
    _, hc = hr.run_and_get_results(input_file, output_file)
    print(f"ending c++ implementation in {time.time() - start_t}")


    print("check")

if __name__ == "__main__":
    main()
