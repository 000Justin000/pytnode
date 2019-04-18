#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "Poisson.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 1, num_params = dim;

	OgataThinning ot(dim);

	Eigen::VectorXd params(num_params);
	params << 0.2;

	Poisson poisson(num_params, dim);
	poisson.SetParameters(params);

	std::vector<Sequence> sequences;

	unsigned num_sequences = 500;
    double T = 100.0;
	std::cout << "1. Simulating " << num_sequences << " sequences with " << T << " length each " << std::endl;

    std::vector<double> vec_T(num_sequences, T);
	ot.Simulate(poisson, vec_T, sequences);

    std::ofstream fout;
    fout.open("data/poisson.csv"); 
    for (Sequence sequence: sequences)
    {
        for (int u = 0; u < 1; u++)
        {
            for (Event evnt: sequence.GetEvents())
                fout << std::fixed << std::setw(12) << std::setprecision(6) << evnt.time;
            fout << ";";
        }
        fout << "\n";
    }

	return 0;
}
