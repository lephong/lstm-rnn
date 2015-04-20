/*
 * Matrix.cpp
 *
 *  Created on: Dec 24, 2014
 *      Author: Phong
 */

#include "Matrix.h"
#include <cblas.h>
#include "Utils.h"
#include <random>

using namespace std;

Matrix::~Matrix() {
    delete[] data;
}

Matrix::Matrix(int rows, int cols) {
	data = new real[rows*cols];
	this->rows = rows;
	this->cols = cols;
	length = rows * cols;
}

// vector
Matrix::Matrix(int rows) {
	data = new real[rows];
	this->rows = rows;
	this->cols = 1;
	length = rows;
}

Matrix::Matrix(int rows, int cols, real* data) {
	this->rows = rows;
	this->cols = cols;
	this->length = rows * cols;
    this->data = new real[length];
    memcpy(this->data, data, sizeof(real)*length);
}

Matrix::Matrix(int rows, real* data) {
	this->rows = rows;
	this->cols = 1;
	this->length = rows;
    this->data = new real[length];
    memcpy(this->data, data, sizeof(real)*length);
}

void Matrix::save(ofstream &f) {
    f << rows << " " << cols << endl;
    for (int i = 0; i < length; i++)
        f << data[i] << " ";
    f << endl;
}

Matrix* Matrix::load(ifstream &f) {
    int rs, cs;
    f >> rs >> cs;
    Matrix* x = new Matrix(rs, cs);
    for (int i = 0; i < x->length; i++)
        f >> x->data[i];
    return x;
}

void Matrix::print(bool trans) {
	if (!trans) {
	    for (int i = 0; i < rows; i++) {
    	    for (int j = 0; j < cols; j++)
        	    cout << data[i*cols + j] << " , ";
	        cout << endl;
    	}
	} 
	else {
		for (int j = 0; j < cols; j++) {
			for (int i = 0; i < rows; i++)
				cout << data[i*cols + j] << " , ";
			cout << endl;
		}
	}
	cout << endl;
}

real Matrix::get(int r, int c) {
	return data[r*cols + c];
}

void Matrix::put(int r, int c, real value) {
	data[r*cols + c] = value;
}

Matrix* Matrix::dup() {
	Matrix* ret = new Matrix(rows, cols);
	memcpy(ret->data, data, sizeof(real)*length);
	return ret;
}

void Matrix::copy(Matrix* other) {
	delete[] data;
	rows = other->rows;
	cols = other->cols;
	length = rows * cols;
	data = new real[length];
	memcpy(data, other->data, sizeof(real)*length);
}

void Matrix::errorIfNotSameSize(Matrix* other) {
	if (rows != other->rows || cols != other->cols)
		Utils::error("size not match");
}

void Matrix::fill(real value) {
	for (int i = 0; i < length; i++)
		data[i] = value;
}

real Matrix::sum() {
	real s = 0;
	for (int i = 0; i < length; i++)
		s += data[i];
	return s;
}

// y <- alpha * x + y
Matrix* Matrix::addi(real alpha, Matrix* x) {
	errorIfNotSameSize(x);
	for (int i = 0; i < length; i++) {
		data[i] += x->data[i] * alpha;
	}
	return this;
}

// y <- x + y
Matrix* Matrix::addi(Matrix* x) {
	return this->addi(1, x);
}


// y[i] <- y[i] + a
Matrix* Matrix::addi(real a) {
	for (int i = 0; i < length; i++) {
		data[i] += a;
	}
	return this;
}

// z <- alpha * x + y
Matrix* Matrix::add(real alpha, Matrix* x) {
	errorIfNotSameSize(x);
	Matrix* z = new Matrix(rows, cols);
	for (int i = 0; i < length; i++) {
		z->data[i] = data[i] + x->data[i] * alpha;
	}
	return z;
}

// y <- x + y
Matrix* Matrix::add(Matrix* x) {
	return this->add(1, x);
}


// z[i] <- y[i] + a
Matrix* Matrix::add(real a) {
	Matrix* z = new Matrix(rows, cols);
	for (int i = 0; i < length; i++) {
		z->data[i] = data[i] + a;
	}
	return z;
}

// z[i] <- x[i] * y[i]
Matrix* Matrix::mul(Matrix* x) {
    errorIfNotSameSize(x);
    Matrix* z = new Matrix(rows, cols);
    for (int i = 0; i < length; i++) {
        z->data[i] = data[i] * x->data[i];
    }
    return z;
}


// y[i] <- x[i] * y[i]
Matrix* Matrix::muli(Matrix* x) {
	errorIfNotSameSize(x);
	for (int i = 0; i < length; i++) {
		data[i] *= x->data[i];
	}
	return this;
}

// y[i] <- a * y[i]
Matrix* Matrix::muli(real a) {
	for (int i = 0; i < length; i++) {
		data[i] *= a;
	}
	return this;
}

// z[i] <- a * y[i]
Matrix* Matrix::mul(real a) {
	Matrix* z = new Matrix(rows, cols);
	for (int i = 0; i < length; i++) {
		z->data[i] = data[i] * a;
	}
	return z;
}

Matrix* Matrix::divi(real a) {
	return this->muli(1/a);
}

Matrix* Matrix::divi(Matrix* x) {
    errorIfNotSameSize(x);
    for (int i = 0; i < length; i++)
        data[i] /= x->data[i];
    return this;
}

Matrix* Matrix::div(real a) {
	return this->mul(1/a);
}

Matrix* Matrix::sqrt() {
    Matrix* z = new Matrix(rows, cols);
    for (int i = 0; i < z->length; i++)
#ifdef __DOUBLE__
        z->data[i] = sqrtl(data[i]);
#else
        z->data[i] = sqrtf(data[i]);
#endif
    return z;
}

Matrix* Matrix::sqrti() {
    for (int i = 0; i < length; i++)
#ifdef __DOUBLE__
        data[i] = sqrtl(data[i]);
#else
        data[i] = sqrtf(data[i]);
#endif
    return this;
}


Matrix* Matrix::getColumn(int col) {
	if (col >= cols) Utils::error("exceed num of cols");

	Matrix* ret = new Matrix(rows);
	for (int i = 0; i < rows; i++) {
		ret->data[i] = data[i*cols + col];
	}
	return ret;
}

Matrix* Matrix::getRows(int* ids, int n) {
	Matrix* ret = new Matrix(n, cols);
	for (int i = 0; i < n; i++) {
		memcpy(&(ret->data[i*cols]), &(data[ids[i]*cols]), sizeof(real)*cols);
	}
	return ret;
}

Matrix* Matrix::addColumn(int c, Matrix* x) {
	if (x->rows != rows) Utils::error("size not match");
	for (int i = 0; i < rows; i++)
		data[i*cols + c] += x->data[i];
	return this;
}

void Matrix::free(Matrix** &As, int n) {
    if (As == NULL) return;
	for (int i = 0; i < n; i++)
		if (As[i] != NULL) delete As[i];
	delete[] As;
    As = NULL;
}

void Matrix::free(Matrix*** &As, int n1, int n2) {
    if (As == NULL) return;
	for (int i1 = 0; i1 < n1; i1++) {
		for (int i2 = 0; i2 < n2; i2++)
			if (As[i1][i2] != NULL) delete As[i1][i2];
		delete[] As[i1];
	}
	delete[] As;
    As = NULL;
}

Matrix* Matrix::zeros(int rows, int cols) {
	Matrix* ret = new Matrix(rows, cols);
	for (int i = 0; i < ret->length; i++)
		ret->data[i] = 0;
	return ret;
}

Matrix* Matrix::zeros(int dim) {
	return zeros(dim, 1);
}

Matrix* Matrix::eye(int dim) {
	Matrix* ret = Matrix::zeros(dim, dim);
	for (int i = 0; i < dim; i++)
		ret->data[i*dim + i] = 1;
	return ret;
}

Matrix* Matrix::uniform(int rows, int cols, real a, real b) {
	Matrix* ret = new Matrix(rows, cols);

	std::uniform_real_distribution<> dis(a, b);
	for (int i = 0; i < ret->length; i++)
		ret->data[i] = dis(randGen);

	return ret;
}

Matrix* Matrix::bernoulli(int rows, int cols, real p) {
	Matrix* ret = Matrix::uniform(rows, cols, 0, 1);
	for (int i = 0; i < ret->length; i++) {
		if (ret->data[i] < p)
			ret->data[i] = 1;
		else
			ret->data[i] = 0;
	}
	return ret;
}

Matrix* Matrix::bernoulli(int rows, real p) {
	return bernoulli(rows, 1, p);
}

Matrix* Matrix::uniform(int rows, real a, real b) {
	return uniform(rows, 1, a, b);
}

Matrix* Matrix::normal(int rows, int cols, real mean, real dev) {
	Matrix* ret = new Matrix(rows, cols);
	default_random_engine generator;
	normal_distribution<double> distribution(mean, dev);

	for (int i = 0; i < ret->length; i++)
		ret->data[i] = distribution(generator);
	return ret;
}

Matrix* Matrix::normal(int rows, real mean, real dev) {
	return normal(rows, 1, mean, dev);
}


void Matrix::gemv(real alpha, enum CBLAS_TRANSPOSE transA, Matrix* A, Matrix* x, real beta, Matrix* y) {
#ifdef __DOUBLE__
	cblas_dgemv(CblasRowMajor, transA, A->rows, A->cols, alpha, A->data, A->cols,
			x->data, 1, beta, y->data, 1);
#else
	cblas_sgemv(CblasRowMajor, transA, A->rows, A->cols, alpha, A->data, A->cols,
			x->data, 1, beta, y->data, 1);
#endif
}

real Matrix::dot(Matrix* X, Matrix* Y) {
#ifdef __DOUBLE__
	return cblas_ddot(X->length, X->data, 1, Y->data, 1);
#else
	return cblas_sdot(X->length, X->data, 1, Y->data, 1);
#endif
}

void Matrix::ger(real alpha, Matrix* x, Matrix* y, Matrix* A) {
#ifdef __DOUBLE__
	cblas_dger(CblasRowMajor, x->length, y->length, alpha, x->data, 1, y->data, 1, A->data, A->cols);
#else
	cblas_sger(CblasRowMajor, x->length, y->length, alpha, x->data, 1, y->data, 1, A->data, A->cols);
#endif
}

void Matrix::axpy(real alpha, Matrix* x, Matrix* y) {
#ifdef __DOUBLE__
	cblas_daxpy(x->length, alpha, x->data, 1, y->data, 1);
#else
	cblas_saxpy(x->length, alpha, x->data, 1, y->data, 1);
#endif

}

real Matrix::nrm2(Matrix* x) {
#ifdef __DOUBLE__
	return cblas_dnrm2(x->length, x->data, 1);
#else
	return cblas_snrm2(x->length, x->data, 1);
#endif
}

