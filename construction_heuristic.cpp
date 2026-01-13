#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include <nlohmann/json.hpp>
#include <CLI/CLI.hpp>

using json = nlohmann::json;

struct HeuristicResult {
    std::vector<std::vector<std::vector<double>>> x;
    std::vector<std::vector<double>> y;
    std::vector<std::vector<std::vector<double>>> w;
    std::vector<std::vector<double>> q;
    std::vector<std::vector<double>> z;
};

HeuristicResult construction_heuristic(int nF, int nC, int nT, 
                                       const std::vector<std::vector<double>>& d, 
                                       const std::vector<double>& q_in) {
    
    // Initialize variables
    std::vector<std::vector<std::vector<double>>> x(nF, std::vector<std::vector<double>>(nC, std::vector<double>(nT, 0.0)));
    std::vector<std::vector<double>> y(nF, std::vector<double>(nT, 1.0));
    std::vector<std::vector<std::vector<double>>> w(nF, std::vector<std::vector<double>>(nF, std::vector<double>(nT, 0.0)));
    std::vector<std::vector<double>> q(nF, std::vector<double>(nT));
    std::vector<std::vector<double>> z(nF, std::vector<double>(nT, 0.0));
    
    // Initialize q by tiling q_in
    for (int f = 0; f < nF; ++f) {
        for (int t = 0; t < nT; ++t) {
            q[f][t] = q_in[f];
        }
    }
    
    // Check for feasibility
    double q_in_sum = std::accumulate(q_in.begin(), q_in.end(), 0.0);
    for (int t = 0; t < nT; ++t) {
        double demand_sum = 0.0;
        for (int c = 0; c < nC; ++c) {
            demand_sum += d[c][t];
        }
        if (q_in_sum < demand_sum) {
            std::cout << "Not enough resources for demand at time " << t 
                      << ". Problem is infeasible." << std::endl;
            std::exit(1);
        }
    }
    
    // Main assignment loop - PARALLELIZED
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < nT; ++t) {
        for (int c = 0; c < nC; ++c) {
            double already_assigned = 0.0;
            
            int f = 0;
            double sum_x_f = 0.0;
            for (int k = 0; k < nC; ++k) {
                sum_x_f += x[f][k][t];
            }
            double residual_capacity = q_in[f] - sum_x_f;
            
            while (already_assigned < d[c][t]) {
                if (residual_capacity > d[c][t] - already_assigned) {
                    x[f][c][t] += d[c][t] - already_assigned;
                    already_assigned = d[c][t];
                } else {
                    if (residual_capacity > 0) {
                        x[f][c][t] += residual_capacity;
                        already_assigned += residual_capacity;
                    }
                    f++;
                    if (f < nF) {
                        sum_x_f = 0.0;
                        for (int k = 0; k < nC; ++k) {
                            sum_x_f += x[f][k][t];
                        }
                        residual_capacity = q_in[f] - sum_x_f;
                    }
                }
            }
        }
    }
    
    // Normalization loop - PARALLELIZED
    // #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < nT; ++t) {
        for (int j = 0; j < nC; ++j) {
            double sum_x_j = 0.0;
            for (int i = 0; i < nF; ++i) {
                sum_x_j += x[i][j][t];
            }
            
            if (sum_x_j != 0.0) {
                for (int i = 0; i < nF; ++i) {
                    x[i][j][t] = x[i][j][t] / sum_x_j;
                }
            }
        }
    }
    
    return {x, y, w, q, z};
}

// Read input from JSON file
bool read_input(const std::string& filename, 
                int& nF, int& nC, int& nT,
                std::vector<std::vector<double>>& d,
                std::vector<double>& q_in) {
    try {
        std::ifstream input_file(filename);
        if (!input_file.is_open()) {
            std::cerr << "Error: Cannot open input file " << filename << std::endl;
            return false;
        }
        
        json j;
        input_file >> j;
        
        nF = j["nF"];
        nC = j["nC"];
        nT = j["nT"];
        
        // Read d (nC x nT matrix)
        d = j["d"].get<std::vector<std::vector<double>>>();
        
        // Read q_in (nF vector)
        q_in = j["q_in"].get<std::vector<double>>();
        
        //std::cout << "Input loaded successfully: nF=" << nF 
        //          << ", nC=" << nC << ", nT=" << nT << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error reading input file: " << e.what() << std::endl;
        return false;
    }
}

// Write output to JSON file
bool write_output(const std::string& filename, const HeuristicResult& result) {
    try {
        json j;
        
        j["x"] = result.x;
        j["y"] = result.y;
        j["w"] = result.w;
        j["q"] = result.q;
        j["z"] = result.z;
        
        std::ofstream output_file(filename);
        if (!output_file.is_open()) {
            std::cerr << "Error: Cannot open output file " << filename << std::endl;
            return false;
        }
        
        // Write with indentation for readability
        output_file << j.dump(2);
        
        // std::cout << "Output written successfully to " << filename << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing output file: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    // Setup CLI
    CLI::App app{"Construction Heuristic Solver"};
    
    // Command-line arguments
    std::string input_file;
    std::string output_file;
    int num_threads = 0;  // 0 means use default/all available
    
    app.add_option("-i,--input", input_file, "Input JSON file path")
        ->required()
        ->check(CLI::ExistingFile);
    
    app.add_option("-o,--output", output_file, "Output JSON file path")
        ->required();
    
    app.add_option("-t,--threads", num_threads, "Number of OpenMP threads (default: all available)")
        ->check(CLI::PositiveNumber);
    
    // Parse command-line arguments
    CLI11_PARSE(app, argc, argv);
    
    // Variables
    int nF, nC, nT;
    std::vector<std::vector<double>> d;
    std::vector<double> q_in;
    
    // Read input
    if (!read_input(input_file, nF, nC, nT, d, q_in)) {
        return 1;
    }
    
    // Set number of threads
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        //std::cout << "Using " << num_threads << " threads" << std::endl;
    } /* else {
        std::cout << "Using default number of threads" << std::endl;
    }*/
    
    // Run heuristic
    // std::cout << "Running construction heuristic..." << std::endl;
    auto result = construction_heuristic(nF, nC, nT, d, q_in);
    
    // Write output
    if (!write_output(output_file, result)) {
        return 1;
    }
    
    // std::cout << "Done!" << std::endl;
    return 0;
}