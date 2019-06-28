#include<cmath> //for std::min/std::max
#include<Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include<RcppThread.h>
// [[Rcpp::depends(RcppThread)]]
#include<RcppParallel.h>
// [[Rcpp::plugins(cpp11)]]

#include "ProgressBar.h"

#define NO_K
#include "PairArithmetic.hpp"

template<typename T>
std::vector< std::vector<T> > noe2_lower(const T* v1, const T* v2, const int n1, const int n2, const int max_idx)
#include "noe2_lower.hpp"

#define ALGORITHM_PARALLEL
template<typename T>
std::vector< std::vector<T> > noe2_lower_p(const T* v1, const T* v2, const int n1, const int n2, const int max_idx)
#include "noe2_lower.hpp"

template<typename T>
std::vector<T> fromNumericVector(const Rcpp::NumericVector& v) {
  return std::vector<T>(v.begin(),v.end());
}

template<typename T>
Rcpp::NumericMatrix toNumericMatrix(const std::vector< std::vector<T> >& m) {
	int n1 = m.size(), n2=m[0].size();
	Rcpp::NumericMatrix ret(n1,n2);
	for(int i=0;i < n1; i++) {
		for(int j=0; j < n2; j++) {
			double d = (double)m[i][j];
			if(d == -1) d = NA_REAL;
			ret(i,j) = d;
		}
	}
	return ret;
}

//' Joint distribution of order statistics for the one- or two-group case
//' 
//' Calculates, under the assumption of joint stochastic independence, the joint distribution of order statistics for the one- or two-group case.
//' 
//' Calculates, under the assumption of joint stochastic independence, the probability P(X_(1:n)<= b_1,...,X_(n:n)<= b_n)
//' where X_(i:n) denotes the i-th order statistic of the sample X_1,...,X_n and the X_k belong to two groups distributed according
//' to a cdf G_1 or G_2.
//' 
//' The implementation is based on a generalization of Noe's recursion.
//' 
//' Returns a matrix M, where M[i,j] is the above probability (for n=i-1+j-1) where (i-1) of the X_k belong to the first group
//' and (j-1) belong to the second group.
//' 
//' @param v1 Boundaries for the first group G_1(b)
//' @param v2 Boundaries for the second group G_2(b)
//' @param n1 The maximal size of the first group - Can be at most length(v1)
//' @param n2 The maximal size of the second group - Can be at most length(v2)
//' @param parallel Use parallelization to speed up the calculation?
//' @export
// [[Rcpp::export]]
Rcpp::NumericMatrix noe_faithful(Rcpp::NumericVector v1,Rcpp::NumericVector v2,int n1=-1,int n2=-1,bool parallel = true,bool progress = true) {
	if(n1<0) n1 = v1.length();
	if(n2<0) n2 = v2.length();
	v1 = Rcpp::rev(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(v1))));
	v2 = Rcpp::rev(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(v2))));
	const int n1_ = std::max(0,std::min((int)v1.length(),n1));
	const int n2_ = std::max(0,std::min((int)v2.length(),n2));
	std::vector< std::vector<PairArithmetic::DoublePair> > res;
	auto fn = [&] {
	  if(parallel) res = noe2_lower_p<PairArithmetic::DoublePair>(fromNumericVector<PairArithmetic::DoublePair>(v1).data(),fromNumericVector<PairArithmetic::DoublePair>(v2).data(),n1_,n2_,std::min(v1.length(),v2.length()));
	  else res = noe2_lower<PairArithmetic::DoublePair>(fromNumericVector<PairArithmetic::DoublePair>(v1).data(),fromNumericVector<PairArithmetic::DoublePair>(v2).data(),n1_,n2_,std::min(v1.length(),v2.length()));
	};
	fn();
	return toNumericMatrix<PairArithmetic::DoublePair>(res);
}