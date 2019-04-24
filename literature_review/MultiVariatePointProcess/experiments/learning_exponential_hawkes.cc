#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <Eigen/Dense>
#include "Sequence.h"
#include "PlainHawkes.h"
#include "utils.h"

int main(const int argc, const char** argv)
{
	std::vector<Sequence> sequences;
    ImportFromFileExistingSequences(argv[1], sequences);

	unsigned dim = 1, num_params = dim * (dim + 1);
	Eigen::MatrixXd beta(dim,dim);
	beta << 1;
	PlainHawkes hawkes(num_params, dim, beta);

	PlainHawkes::OPTION options;
	options.method = PlainHawkes::PLBFGS;
	options.base_intensity_regularizer = PlainHawkes::NONE;
	options.excitation_regularizer = PlainHawkes::NONE;
	hawkes.fit(sequences, options);
	
	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << hawkes.GetParameters().transpose() << std::endl;

	return 0;
}
