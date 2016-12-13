# Cholesky Factorization Makefile

# Specify path to Eigen library
INC = /usr/include/eigen3/

# Specify compiler
COMP = g++ -I $(INC) -std=c++1y -O2

# Specify target
all: simplex_test

# Build executable
simplex_test: simplex_test.cpp
	$(COMP) simplex_test.cpp -o simplex_test

# Clean build
clean:
	rm simplex_test
