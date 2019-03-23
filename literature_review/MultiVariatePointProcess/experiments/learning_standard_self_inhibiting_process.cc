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

	std::vector<double> vec_T(500, 100.0);

	OgataThinning ot(dim);
	ot.Simulate(inhibiting, vec_T, sequences);

	SelfInhibitingProcess::OPTION options;
	options.base_intensity_regularizer = SelfInhibitingProcess::NONE;
	options.excitation_regularizer = SelfInhibitingProcess::NONE;
	options.coefficients[SelfInhibitingProcess::LAMBDA0] = 0;
	options.coefficients[SelfInhibitingProcess::LAMBDA] = 0;

	SelfInhibitingProcess inhibiting_new(num_params, dim);

	inhibiting_new.fit(sequences, options);

	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << inhibiting_new.GetParameters().transpose() << std::endl;
	std::cout << "True Parameters : " << std::endl;
	std::cout << params.transpose() << std::endl;

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
