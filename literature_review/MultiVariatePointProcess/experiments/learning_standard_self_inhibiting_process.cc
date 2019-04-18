#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <Eigen/Dense>
#include "Sequence.h"
#include "SelfInhibitingProcess.h"
#include "utils.h"

int main(const int argc, const char** argv)
{
	std::vector<Sequence> sequences;
    ImportFromFileExistingSequences(argv[1], sequences);

	unsigned dim = 1, num_params = dim * (dim + 1);
	SelfInhibitingProcess inhibiting(num_params, dim);

	SelfInhibitingProcess::OPTION options;
	options.base_intensity_regularizer = SelfInhibitingProcess::NONE;
	options.excitation_regularizer = SelfInhibitingProcess::NONE;
	options.coefficients[SelfInhibitingProcess::LAMBDA0] = 0;
	options.coefficients[SelfInhibitingProcess::LAMBDA] = 0;
	inhibiting.fit(sequences, options);

	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << inhibiting.GetParameters().transpose() << std::endl;

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
