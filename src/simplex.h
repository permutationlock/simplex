/* simplex.h
 * Aven Bross (dabross@alaska.edu)
 * 
 * Eigen Simplex implementation
 */

#ifndef __SIMPLEX_H__
#define __SIMPLEX_H__

#define SIMPLEX_VERBOSE

#include <iostream>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

// Takes a linear program and a basic feasible point 
// Modifes bfp according to simplex algorithm until:
//     1) bfp is a solution; return true
//     2) problem is unbounded; return false
template<typename matrix_t, typename vector_t>
bool simplex_phase_ii(
		std::vector<std::size_t> & b_indices, std::vector<std::size_t> n_indices,
		vector_t & c, matrix_t & a, vector_t & b
	)
{
	std::size_t m = a.rows(), n = a.cols();
	
	#ifdef SIMPLEX_VERBOSE
		std::cout << "min c^Tx, x in R^" << n << " s.t. Ax = b, x>=0\n";
		std::cout << "c^T = " << c.transpose() << "\n";
		std::cout << "A =\n" << a << "\n";
		std::cout << "b^T = " << b.transpose() << "\n";
	#endif
	
	if(b_indices.size() != m || n_indices.size() != n-m)
		throw std::logic_error("index sets size incorrect");
	
	while(1) {
		Eigen::MatrixXd b_matrix(m,m);
		Eigen::MatrixXd c_b(m,1);
		for(std::size_t i = 0; i < m; ++i) {
			b_matrix.col(i) = a.col(b_indices[i]);
			c_b(i) = c(b_indices[i]);
		}
		
		Eigen::MatrixXd n_matrix(m,n-m);
		Eigen::MatrixXd c_n(n-m,1);
		for(std::size_t i = 0; i < n-m; ++i) {
			n_matrix.col(i) = a.col(n_indices[i]);
			c_n(i) = c(n_indices[i]);
		}
		
		#ifdef SIMPLEX_VERBOSE
			std::cout << "B =\n" << b_matrix << "\n";
			std::cout << "N =\n" << n_matrix << "\n";
			std::cout << "c_B^T = " << c_b.transpose() << "\n";
			std::cout << "c_N^T = " << c_n.transpose() << "\n";
		#endif
		
		
		Eigen::MatrixXd x_b = b_matrix.fullPivLu().solve(b);
		Eigen::MatrixXd y = b_matrix.transpose().fullPivLu().solve(c_b);
		Eigen::MatrixXd s_n = c_n - n_matrix.transpose() * y;
		
		#ifdef SIMPLEX_VERBOSE
			std::cout << "x_B^T = " << x_b.transpose() << "\n";
			std::cout << "y^T = " << y.transpose() << "\n";
			std::cout << "s_N^T = " << s_n.transpose() << "\n";
		#endif
	
		int q = -1;
		double min = 0.0;
	
		for(std::size_t i = 0; i < n-m; ++i) {
			if(s_n(i) < min) {
				q = n_indices[i];
				min = s_n(i);
			}
		}
	
		if(q == -1) {
			#ifdef SIMPLEX_VERBOSE
				std::cout << "SOLUTION FOUND!\n";
			#endif
			return true;
		}
		
		#ifdef SIMPLEX_VERBOSE
			std::cout << "q = " << q << "\n";
		#endif
	
		Eigen::MatrixXd d = b_matrix.fullPivLu().solve(a.col(q));
		
		#ifdef SIMPLEX_VERBOSE
			std::cout << "d = " << d.transpose() << "\n";
		#endif
	
		int p = -1;
		min = 0.0;
	
		for(std::size_t i = 0; i < m; ++i) {
			if(d(i) > 0) {
				double val = x_b(i) / d(i);
				if(val < min || p == -1) {
					p = b_indices[i];
					min = val;
				}
			}
		}
	
		if(p == -1) {
			#ifdef SIMPLEX_VERBOSE
				std::cout << "PROBLEM UNBOUNDED!\n";
			#endif
			return false;
		}
		
		#ifdef SIMPLEX_VERBOSE
			std::cout << "p = " << p << "\n";
		#endif
	
		for(std::size_t i = 0; i < m; ++i) {
			if(b_indices[i] == p) {
				b_indices[i] = q;
				break;
			}
		}
	
		for(std::size_t i = 0; i < n-m; ++i) {
			if(n_indices[i] == q) {
				n_indices[i] = p;
				break;
			}
		}
		
		#ifdef SIMPLEX_VERBOSE
			std::cout << "added q removed p\n\n";
		#endif
	}
}

#endif

