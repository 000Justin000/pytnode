#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <Eigen/Dense>
#include "Sequence.h"
#include "Poisson.h"
#include "utils.h"

int main(const int argc, const char** argv)
{
	std::vector<Sequence> sequences;
    ImportFromFileExistingSequences(argv[1], sequences);

	unsigned dim = 1, num_params = dim;

	Poisson poisson(num_params, dim);

	std::cout << "2. Fitting Parameters " << std::endl << std::endl;  
	poisson.fit(sequences);

	std::cout << "Estimated Parameters : " << std::endl;
	std::cout << poisson.GetParameters().transpose() << std::endl;

	return 0;
}
