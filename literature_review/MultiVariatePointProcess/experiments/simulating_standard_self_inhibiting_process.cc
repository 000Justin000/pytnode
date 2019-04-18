#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "SelfInhibitingProcess.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 1;
	unsigned num_params = dim * (dim + 1);

	Eigen::VectorXd params(num_params);
	params << 0.50, 0.20;

	std::vector<Sequence> sequences;

	SelfInhibitingProcess inhibiting(num_params, dim);
	inhibiting.SetParameters(params);

	unsigned num_sequences = 500;
    double T = 100.0;
	std::vector<double> vec_T(num_sequences, T);
	std::cout << "1. Simulating " << num_sequences << " sequences with " << T << " length each " << std::endl;

	OgataThinning ot(dim);
	ot.Simulate(inhibiting, vec_T, sequences);

    std::ofstream fout;
    fout.open("data/self_inhibiting.csv"); 
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
