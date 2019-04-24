#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "PlainHawkes.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 1, num_params = dim * (dim + 1);

	OgataThinning ot(dim);

	Eigen::VectorXd params(num_params);
	params << 0.2, 0.8;

	Eigen::MatrixXd beta(dim,dim);
	beta << 1;

	PlainHawkes hawkes(num_params, dim, beta);
	hawkes.SetParameters(params);

	std::vector<Sequence> sequences;

	unsigned num_sequences = 500;
    double T = 100.0;
	std::cout << "1. Simulating " << num_sequences << " sequences with " << T << " length each " << std::endl;

    std::vector<double> vec_T(num_sequences, T);

	ot.Simulate(hawkes, vec_T, sequences);

    std::ofstream fout;
    fout.open("data/exponential_hawkes.csv"); 
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
