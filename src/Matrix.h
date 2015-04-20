/*
 * Matrix.h
 *
 *  Created on: Dec 24, 2014
 *      Author: Phong
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include "Default.h"
#include <cblas.h>
#include <fstream>
#include <random>

using namespace std;

// row-major

class Matrix {
public:
	int rows, cols, length;
	real* data;

    ~Matrix();
    
	Matrix(int rows, int cols);

	// vector
	Matrix(int rows);

	Matrix(int rows, int cols, real* data);

	Matrix(int rows, real* data);

	real get(int r, int c);

	void put(int r, int c, real value);

	Matrix* dup();

	void copy(Matrix* other);

	void errorIfNotSameSize(Matrix* other);

	void fill(real value);
    
    void save(ofstream& f);
    
    static Matrix* load(ifstream& f);
    
    void print(bool trans = false);

	// y <- alpha * x + y
	Matrix* addi(real alpha, Matrix* x);

	// y <- x + y
	Matrix* addi(Matrix* x);

	// y[i] <- y[i] + a
	Matrix* addi(real a);

	// z <- alpha * x + y
	Matrix* add(real alpha, Matrix* x);

	// y <- x + y
	Matrix* add(Matrix* x);

	// z[i] <- y[i] + a
	Matrix* add(real a);

    // z[i] <- x[i] * y[i]
    Matrix* mul(Matrix* x);
    
	// y[i] <- x[i] * y[i]
	Matrix* muli(Matrix* x);

	// y[i] <- a * y[i]
	Matrix* muli(real a);

	// z[i] <- a * y[i]
	Matrix* mul(real a);

	Matrix* divi(real a);
    
    Matrix* divi(Matrix* x);

	Matrix* div(real a);
    
    Matrix* sqrt();

    Matrix* sqrti();

	Matrix* getColumn(int col);

	Matrix* getRows(int* ids, int n);

	Matrix* addColumn(int c, Matrix* x);

	real sum();

	static void free(Matrix** &As, int n);

	static void free(Matrix*** &As, int n1, int n2);

	static Matrix* zeros(int rows, int cols);

	static Matrix* zeros(int dim);

	static Matrix* eye(int dim);

	static Matrix* uniform(int rows, int cols, real a, real b);

	static Matrix* uniform(int rows, real a, real b);

	static Matrix* bernoulli(int rows, int cols, real p);

	static Matrix* bernoulli(int rows, real p);

	static Matrix* normal(int rows, int cols, real mean, real dev);

	static Matrix* normal(int rows, real mean, real dev);

	static void gemv(real alpha, enum CBLAS_TRANSPOSE transA, Matrix* A, Matrix* x, real beta, Matrix* y);

	static real dot(Matrix* X, Matrix* Y);

	static void ger(real alpha, Matrix* x, Matrix* y, Matrix* A);

	static void axpy(real alpha, Matrix* x, Matrix* y);

	static real nrm2(Matrix* x);

};

#endif /* MATRIX_H_ */
