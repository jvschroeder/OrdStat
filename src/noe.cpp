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

#undef ALGORITHM_PARALLEL
#define PROGRESS
template<typename T>
std::vector< std::vector<T> > noe2_lower_progress(const T* v1, const T* v2, const int n1, const int n2, const int max_idx)
#include "noe2_lower.hpp"
  
#define ALGORITHM_PARALLEL
  template<typename T>
  std::vector< std::vector<T> > noe2_lower_p_progress(const T* v1, const T* v2, const int n1, const int n2, const int max_idx)
#include "noe2_lower.hpp"

template<typename T>
std::vector<T> fromNumericVector(const Rcpp::NumericVector& v) {
  return std::vector<T>(v.begin(),v.end());
}

template<typename T>
Rcpp::NumericVector toNumericVector(const std::vector<T>& v) {
  return Rcpp::NumericVector(v.begin(),v.end());
}

Rcpp::NumericVector toNumericVector(const std::vector< PairArithmetic::DoublePair >& v) {
  std::vector<double> v1(v.size());
  std::transform(v.begin(),v.end(),v1.begin(),[](const PairArithmetic::DoublePair& v){ return (double) v; });
  return Rcpp::NumericVector(v1.begin(),v1.end());
}

template<typename T>
struct fd_fm {
  const std::vector<int> R,V;
  const std::vector<T> p;
};

template<typename T>
Rcpp::DataFrame toDataFrame(const fd_fm<T>& d) {
  return Rcpp::DataFrame::create( Rcpp::Named("R") = toNumericVector(d.R) ,
   Rcpp::Named("V") = toNumericVector(d.V),
   Rcpp::Named("p") = toNumericVector(d.p) );
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

template<typename T>
std::vector< std::vector<T> > noe_faithful(const T* v1, const T* v2, int n1,int n2,bool parallel, bool progress, const int max_idx) {
  std::vector< std::vector<T> > res(0);
  auto fn = [&] {
    if(parallel) {
      if(!progress) res = noe2_lower_p<T>(v1,v2,n1,n2,max_idx);
      else res = noe2_lower_p_progress<T>(v1,v2,n1,n2,max_idx);
    } else{
      if(!progress) res = noe2_lower<T>(v1,v2,n1,n2,max_idx);
      else res = noe2_lower_progress<T>(v1,v2,n1,n2,max_idx);
    }
  };
  try{
    fn();
  } catch(const RcppThread::UserInterruptException& ex) {
    if(progress )Rcpp::Rcout << std::endl;
    throw ex;
  }
  if(progress) Rcpp::Rcout << std::endl;
  return res;
}

//' Joint distribution of order statistics for the two-group case
//' 
//' Calculates, under the assumption of joint stochastic independence, the joint distribution of order statistics for the one- or two-group case.
//' 
//' Calculates, under the assumption of joint stochastic independence, the probability P(X_(1:n)<= b_1,...,X_(n:n)<= b_n)
//' where X_(i:n) denotes the i-th order statistic of the sample X_1,...,X_n and the X_k belong to two groups.
//' The first is Uni[0,1] and the seconds is given by its cdf Fn_alt. 
//' 
//' The implementation is based on a generalization of Noe's recursion. For the details see https://arxiv.org/abs/1812.09063.
//' 
//' @param b Vector of upper bounds b
//' @param Fn_alt Distribution (cdf) under the alternative
//' @param n1 The maximal size of the first group - Can be at most length(b)
//' @param n2 The maximal size of the second group - Can be at most length(b)
//' @param parallel Use parallelization to speed up the calculation?
//' @param progress Show a progress bar?
//' @param quick Use lower precision to speed up calculation?
//' 
//' @return Returns a matrix M, where M[i,j] is the above probability (for n=i-1+j-1) where (i-1) of the X_k belong to the first group
//' and (j-1) belong to the second group.
//'
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::NumericMatrix pordstat2(Rcpp::NumericVector b, Rcpp::Function Fn_alt, int n1=-1, int n2=-1, bool parallel = true, bool progress = true, bool quick = false) {
	Rcpp::NumericVector upper1 = b;
	Rcpp::NumericVector upper2 = Rcpp::as<Rcpp::NumericVector>(Fn_alt(b));
	if(n1<0) n1 = upper1.length();
	if(n2<0) n2 = upper2.length();
	upper1 = Rcpp::rev(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(upper1))));
	upper2 = Rcpp::rev(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(upper2))));
	const int n1_ = std::max(0,std::min((int)upper1.length(),n1));
	const int n2_ = std::max(0,std::min((int)upper2.length(),n2));
	if(!quick) {
	  std::vector< std::vector<PairArithmetic::DoublePair> > res = noe_faithful<PairArithmetic::DoublePair>(fromNumericVector<PairArithmetic::DoublePair>(upper1).data(),fromNumericVector<PairArithmetic::DoublePair>(upper2).data(),n1_,n2_,parallel,progress,std::min(upper1.length(),upper2.length()));
	  return toNumericMatrix<PairArithmetic::DoublePair>(res);
	} else {
	  std::vector< std::vector<double> > res = noe_faithful<double>(fromNumericVector<double>(upper1).data(),fromNumericVector<double>(upper2).data(),n1_,n2_,parallel,progress,std::min(upper1.length(),upper2.length()));
	  return toNumericMatrix<double>(res);
	}
}

template<typename T>
fd_fm<T> fd_fm_(const std::vector< std::vector<T> >& order_stat,const Rcpp::NumericVector& v1,const Rcpp::NumericVector& v2,const int m0,const int m) {
  std::vector<int> R(0),V(0);
  std::vector<T> p(0);
  // *Rf_choose(m0,j)
  T c1 = 1;
  for(int j = 0; j <= m0; j++) {
    // *Rf_choose(m-m0,k-j)
    T c2 = 1;
    for(int k = j; k <= m; k++) {
      if(j>=k-m+m0) {
        T val = c1 * c2 * order_stat[m0-j][m-k-(m0-j)]; 
        if(k>0) val *= PairArithmetic::pow(T(1-v1[m-k]),j) * PairArithmetic::pow(T(1-v2[m-k]),k-j);
        R.push_back(k);
        V.push_back(j);
        p.push_back(val);
        c2 *= m-m0-(k-j);
        c2 /= k-j+1;
      }
    }
    c1 *= m0-j;
    c1 /= j+1;
  }
  fd_fm<T> d = { R, V, p };
  return d;
}

//' Calculates the joint density of the number discoveries R and the number of false discoveries V
//' 
//' Calculates, under the assumption of joint stochastic independence, the joint density of the number discoveries R and the number of false discoveries V of
//' an FDR controlling step-up test given by the vector of thresholds b.
//' 
//' Calculates, under the assumption
//' * of joint stochastic independence and
//' * that the p-values are Uni[0,1] under H_0 and distributed according to Fn_alt under H_1 and
//' * that exactly m0 of the m hypotheses are true
//' the joint density of the number discoveries R and the number of false discoveries V of
//' an FDR controlling step-up test given by the vector of thresholds b. This corresponds to
//' the model FM(m,m0,Fn_alt) in https://arxiv.org/abs/1812.09063.
//' 
//' The implementation is based on a generalization of Noe's recursion. For the details see https://arxiv.org/abs/1812.09063.
//' 
//' @param b Vector of upper bounds b
//' @param Fn_alt Distribution (cdf) under the alternative
//' @param m0 The number of true hypotheses - Can be at most m
//' @param m The number of hypotheses - Can be at most length(b)
//' @param parallel Use parallelization to speed up the calculation?
//' @param progress Show a progress bar?
//' @param quick Use lower precision to speed up calculation?
//' 
//' @return Returns a data.frame with three columns. The first column 'R' is the number of discoveries, the second column 'V' is the number
//' of false discoveries and the third column gives the joint probability of making exactly this number of discoveries and false discoveries.
//' 
//' @examples
//' #Equation (24) in Glueck, D., Mandel, J., Karimpour-Fard, A., et al. (2008). Exact Calculations of Average Power for the Benjamini-Hochberg Procedure. The International Journal of Biostatistics, 4(1), doi:10.2202/1557-4679.1103
//' Fn <- function(t) 1 + pnorm(qnorm(t/2)-sqrt(N)) - pnorm(qnorm(1-t/2)-sqrt(N))
//' #Setting of Table 2 of Glueck (2008)
//' N <- 5
//' m <- 5
//' m0 <- 4
//' t <- 0.05 * (1:m)/m
//' density <- OrdStat::dfd_fm(t,Fn,m0,m,progress=F)
//' #Calculate the last power value in Table 2 of Glueck (2008)
//' print(with(density,sum(p*(R-V)/(m-m0))))
//' 
//' @export
// [[Rcpp::export(rng = false)]]
Rcpp::DataFrame dfd_fm(Rcpp::NumericVector b, Rcpp::Function Fn_alt, const int m0, const int m, bool parallel = true, bool progress = true, bool quick = false) {
  Rcpp::NumericVector upper1 = Rcpp::clone(b);
  Rcpp::NumericVector upper2 = Rcpp::as<Rcpp::NumericVector>(Fn_alt(b));
  upper1 = 1.0-(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(upper1))));
  upper2 = 1.0-(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(upper2))));
  if(upper1.length() < m) ::Rf_error("upper1 too short!");
  if(!quick) {
    std::vector< std::vector<PairArithmetic::DoublePair> > res = noe_faithful<PairArithmetic::DoublePair>(fromNumericVector<PairArithmetic::DoublePair>(upper1).data(),fromNumericVector<PairArithmetic::DoublePair>(upper2).data(),m0,m-m0,parallel,progress,std::min(upper1.length(),upper2.length()));
    return toDataFrame(fd_fm_<PairArithmetic::DoublePair>(res,upper1,upper2,m0,m));
  } else {
    std::vector< std::vector<double> > res = noe_faithful<double>(fromNumericVector<double>(upper1).data(),fromNumericVector<double>(upper2).data(),m0,m-m0,parallel,progress,std::min(upper1.length(),upper2.length()));
    return toDataFrame(fd_fm_<double>(res,upper1,upper2,m0,m));
  }
}

//' Calculates the average power of a FDR controlling procedure
//' 
//' Calculates, under the assumption of joint stochastic independence, the average power
//' E[(R-V)/(m-m_0)] of a FDR controlling step-up test given by the vector of thresholds b.
//' 
//' Calculates, under the model FM(m,m0,Fn_alt) in https://arxiv.org/abs/1812.09063 the average power E[(R-V)/(m-m_0)].
//' For the definition of R and V see the documentation of \code{dfd_fm}.
//' 
//' The implementation is based on a generalization of Noe's recursion. For the details see https://arxiv.org/abs/1812.09063.
//' 
//' @param b Vector of upper bounds b
//' @param Fn_alt Distribution (cdf) under the alternative
//' @param m0 The number of true hypotheses - Can be at most m
//' @param m The number of hypotheses - Can be at most length(b)
//' @param parallel Use parallelization to speed up the calculation?
//' @param progress Show a progress bar?
//' @param quick Use lower precision to speed up calculation?
//' 
//' @return Returns the average power
//' 
//' @examples
//' #Equation (24) in Glueck, D., Mandel, J., Karimpour-Fard, A., et al. (2008). Exact Calculations of Average Power for the Benjamini-Hochberg Procedure. The International Journal of Biostatistics, 4(1), doi:10.2202/1557-4679.1103
//' Fn <- function(t) 1 + pnorm(qnorm(t/2)-sqrt(N)) - pnorm(qnorm(1-t/2)-sqrt(N))
//' #Setting of Table 2 of Glueck (2008)
//' N <- 5
//' m <- 5
//' m0 <- 4
//' t <- 0.05 * (1:m)/m
//' #Calculate the last power value in Table 2 of Glueck (2008)
//' print(OrdStat::avg_pwr(t,Fn,m0,m,progress=F))
//' 
//' @export
// [[Rcpp::export(rng = false)]]
double avg_pwr(Rcpp::NumericVector b, Rcpp::Function Fn_alt, const int m0, const int m, bool parallel = true, bool progress = true, bool quick = false) {
  Rcpp::NumericVector upper1 = Rcpp::clone(b);
  Rcpp::NumericVector upper2 = Rcpp::as<Rcpp::NumericVector>(Fn_alt(b));
  upper1 = 1.0-(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(upper1))));
  upper2 = 1.0-(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(upper2))));
  if(upper1.length() < m) ::Rf_error("upper1 too short!");
  if(!quick) {
    std::vector< std::vector<PairArithmetic::DoublePair> > res = noe_faithful<PairArithmetic::DoublePair>(fromNumericVector<PairArithmetic::DoublePair>(upper1).data(),fromNumericVector<PairArithmetic::DoublePair>(upper2).data(),m0,m-m0,parallel,progress,std::min(upper1.length(),upper2.length()));
    fd_fm<PairArithmetic::DoublePair> d = fd_fm_<PairArithmetic::DoublePair>(res,upper1,upper2,m0,m);
    PairArithmetic::DoublePair result = 0;
    for(size_t i = 0; i < d.p.size(); i++) {
      PairArithmetic::DoublePair val = d.p[i];
      val *= (d.R[i]-d.V[i]);
      val /= (double)(m-m0);
      result += val;
    }
    return (double)result;
  } else {
    std::vector< std::vector<double> > res = noe_faithful<double>(fromNumericVector<double>(upper1).data(),fromNumericVector<double>(upper2).data(),m0,m-m0,parallel,progress,std::min(upper1.length(),upper2.length()));
    fd_fm<double> d = fd_fm_<double>(res,upper1,upper2,m0,m);
    PairArithmetic::DoublePair result = 0;
    for(size_t i = 0; i < d.p.size(); i++) {
      PairArithmetic::DoublePair val = d.p[i];
      val *= (d.R[i]-d.V[i]);
      val /= (double)(m-m0);
      result += val;
    }
    return (double)result;
  }
}

//' Calculates the lambda-power of a FDR controlling procedure
//' 
//' Calculates, under the model RM(m,pr,Fn_alt) in https://arxiv.org/abs/1812.09063 the lambda-power P[(R-V)/(m-M_0)>=lambda].
//' 
//' Calculates, under the assumption of joint stochastic independence, the lambda-power
//' P[(R-V)/(m-M_0)>=lambda] of a FDR controlling step-up test given by the vector of thresholds b
//' and where M_0 is binomially distributed pbinom(.,m,pr).
//' For the definition of R and V see the documentation of \code{dfd_fm}.
//' 
//' The implementation is based on a generalization of Noe's recursion. For the details see https://arxiv.org/abs/1812.09063.
//' 
//' @param b Vector of upper bounds b
//' @param Fn_alt Distribution (cdf) under the alternative
//' @param pr The probability that a given hypothesis is true
//' @param m The number of hypotheses - Can be at most length(b)
//' @param lambda The threshold lambda
//' @param parallel Use parallelization to speed up the calculation?
//' @param progress Show a progress bar?
//' @param quick Use lower precision to speed up calculation?
//' 
//' @return Returns the lambda power
//' 
//' @examples
//' #Calculate the lambda power for the setting given by the first column in Table 3
//' # of Izmirlian, G., (2018). Average power and lambda-power in multiple testing
//' # scenarios when the benjamini-hochberg false discovery rate procedure is used. arXiv preprint arXiv:1801.03989
//' lambda <- 0.9
//' m <- 200
//' n <- 70
//' r <- 5/200
//' alpha <- 0.15
//' theta <- 0.6
//' df <- 2*n-2
//' ncp <- sqrt(n/2) * theta
//' t <- alpha * (1:m)/m
//' pr <- 1-r
//' Fn <- function(t) pt(qt(t/2,df,0,F),df,ncp,F)-pt(-qt(t/2,df,0,F),df,ncp)
//' print(OrdStat::lambda_pwr(t,Fn,pr,m,lambda))
//' 
//' @export
// [[Rcpp::export(rng = false)]]
double lambda_pwr(Rcpp::NumericVector b, Rcpp::Function Fn_alt, const double pr, const int m, const double lambda, bool parallel = true, bool progress = true, bool quick = false) {
  Rcpp::NumericVector upper1 = Rcpp::clone(b);
  Rcpp::NumericVector upper2 = Rcpp::as<Rcpp::NumericVector>(Fn_alt(b));
  upper1 = 1.0-(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(upper1))));
  upper2 = 1.0-(Rcpp::NumericVector(Rcpp::cummin(Rcpp::rev(upper2))));
  if(upper1.length() < m) ::Rf_error("upper1 too short!");
  if(!quick) {
    std::vector< std::vector<PairArithmetic::DoublePair> > res = noe_faithful<PairArithmetic::DoublePair>(fromNumericVector<PairArithmetic::DoublePair>(upper1).data(),fromNumericVector<PairArithmetic::DoublePair>(upper2).data(),m,std::ceil((1-lambda)*m),parallel,progress,std::min(upper1.length(),upper2.length()));
    PairArithmetic::DoublePair result = 0;
    for(int m0 = 0; m0 <= m; m0++) {
      fd_fm<PairArithmetic::DoublePair> d = fd_fm_<PairArithmetic::DoublePair>(res,upper1,upper2,m0,m);
      for(size_t i = 0; i < d.p.size(); i++) {
        if((d.R[i]-d.V[i])>=lambda*(m-m0))
        result += PairArithmetic::fast_pow(PairArithmetic::DoublePair(pr),m0) * PairArithmetic::fast_pow(PairArithmetic::DoublePair(1-pr),m-m0) * PairArithmetic::choose<PairArithmetic::DoublePair>(m,m0) * d.p[i];
      }
    }
    return (double)result;
  } else {
    std::vector< std::vector<double> > res = noe_faithful<double>(fromNumericVector<double>(upper1).data(),fromNumericVector<double>(upper2).data(),m,std::ceil((1-lambda)*m),parallel,progress,std::min(upper1.length(),upper2.length()));
    PairArithmetic::DoublePair result = 0;
    for(int m0 = 0; m0 <= m; m0++) {
      fd_fm<double> d = fd_fm_<double>(res,upper1,upper2,m0,m);
      for(size_t i = 0; i < d.p.size(); i++) {
        if((d.R[i]-d.V[i])>=lambda*(m-m0))
          result += PairArithmetic::fast_pow(PairArithmetic::DoublePair(pr),m0) * PairArithmetic::fast_pow(PairArithmetic::DoublePair(1-pr),m-m0) * PairArithmetic::choose<PairArithmetic::DoublePair>(m,m0) * d.p[i];
      }
    }
    return (double)result;
  }
}