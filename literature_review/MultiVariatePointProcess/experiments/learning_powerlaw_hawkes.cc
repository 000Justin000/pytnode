#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <Eigen/Dense>
#include "Sequence.h"
#include "HawkesGeneralKernel.h"
#include "PowerlawKernel.h"
#include "utils.h"

int main(const int argc, const char** argv)
{
	std::vector<Sequence> sequences;
    ImportFromFileExistingSequences(argv[1], sequences);

	unsigned dim = 1, num_params = dim * (dim + 1);
	std::vector<std::vector<TriggeringKernel*> > triggeringkernels(dim, std::vector<TriggeringKernel*>(dim, NULL));
	for(unsigned m = 0; m < dim; ++ m)
	{
		for(unsigned n = 0; n < dim; ++ n)
		{
            triggeringkernels[m][n] = new PowerlawKernel(2.0, 1.0);
		}
	}
	HawkesGeneralKernel hawkes(num_params, dim, triggeringkernels);

	HawkesGeneralKernel::OPTION options;
	options.method = HawkesGeneralKernel::PLBFGS;
	options.base_intensity_regularizer = HawkesGeneralKernel::NONE;
	options.excitation_regularizer = HawkesGeneralKernel::NONE;
	hawkes.fit(sequences, options);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << hawkes.GetParameters().transpose() << std::endl;

    std::ofstream fout;
    fout.open("simulation.csv"); 
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
