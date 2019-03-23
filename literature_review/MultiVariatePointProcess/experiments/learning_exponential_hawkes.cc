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

	PlainHawkes hawkes_new(num_params, dim, beta);
	PlainHawkes::OPTION options;
	options.method = PlainHawkes::PLBFGS;
	options.base_intensity_regularizer = PlainHawkes::NONE;
	options.excitation_regularizer = PlainHawkes::NONE;

	std::cout << "2. Fitting Parameters " << std::endl << std::endl;  
	hawkes_new.fit(sequences, options);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << hawkes_new.GetParameters().transpose() << std::endl;
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
