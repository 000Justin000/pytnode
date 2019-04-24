#include <iostream>
#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "HawkesGeneralKernel.h"
#include "PowerlawKernel.h"

int main(const int argc, const char** argv)
{
	unsigned dim = 1, num_params = dim * (dim + 1);
	Eigen::VectorXd params(num_params);
    params << 0.2, 0.8;

	std::vector<std::vector<TriggeringKernel*> > triggeringkernels(dim, std::vector<TriggeringKernel*>(dim, NULL));

	for(unsigned m = 0; m < dim; ++ m)
	{
		for(unsigned n = 0; n < dim; ++ n)
		{
            triggeringkernels[m][n] = new PowerlawKernel(2.0, 1.0);
		}
	}

	HawkesGeneralKernel hawkes(num_params, dim, triggeringkernels);
	hawkes.SetParameters(params);
	OgataThinning ot(dim);

	unsigned num_sequences = 500;
    double T = 100.0;
	std::cout << "1. Simulating " << num_sequences << " sequences with " << T << " length each " << std::endl;

    std::vector<double> vec_T(num_sequences, T);

	std::vector<Sequence> sequences;
	ot.Simulate(hawkes, vec_T, sequences);

    std::ofstream fout;
    fout.open("data/powerlaw_hawkes.csv"); 
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
