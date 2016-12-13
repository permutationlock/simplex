/* simplex_test.cpp
 * Aven Bross dabross@alaska.edu
 * Testing simplex code.
 */

#include <iostream>
#include <stdexcept>
#include <Eigen/Dense>
#include "simplex.h"

int main() {
	Eigen::MatrixXd a(2,4);
	Eigen::MatrixXd c(4,1);
	Eigen::MatrixXd b(2,1);
	c << 10, -2,  0,  0;
	a << -3,  1,  1,  0,
		  6, -2,  0,  1;
	b <<  1,  9;
	
	std::vector<std::size_t> b_indices = {2,3};
	std::vector<std::size_t> n_indices = {0,1};
	
	simplex_phase_ii(b_indices, n_indices, c, a, b);
}
