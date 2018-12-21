//#ifdef ALGORITHM_PARALLEL
//#include "tbb/enumerable_thread_specific.h"
//#include "tbb/parallel_for.h"
//#endif
//#include<vector>

{
	#ifdef ALGORITHM_PARALLEL
	tbb::enumerable_thread_specific< std::vector<T> > coeffs1_(std::vector<T>(n1+1,-1));
	tbb::enumerable_thread_specific< std::vector<T> > coeffs2_(std::vector<T>(n2+1,-1));
		#ifdef LOGN_SUMMATION
		tbb::enumerable_thread_specific< std::vector<T> > sum_((n1+1)*(n2+1));
		#endif
	#else
	std::vector<T> coeffs1(n1+1,-1);
	std::vector<T> coeffs2(n2+1,-1);
		#ifdef LOGN_SUMMATION
		std::vector<T> sum((n1+1)*(n2+1));
		#endif
	#endif
	std::vector< std::vector<T> > res(n1+1, std::vector<T>(n2+1,-1));
	std::vector< std::vector<T> > Q(res), Q_new(res);
	T val = 1;
	for(int i1=0; i1<=n1; i1++) {
		T val_ = val;
		for(int i2=0; i2<=n2; i2++) {
			Q[i1][i2] = val_;
			val_ *= v2[0];
		}
		val *= v1[0];
	}
	res[0][0] = 1;
	if(n1>0) res[1][0] = v1[0];
	if(n2>0) res[0][1] = v2[0];
	const int n = n1+n2;
	for(int m = 2; m <= std::min(n,max_idx); m++) {
		const T d1 = v1[m-1]-v1[m-2];
		const T d2 = v2[m-1]-v2[m-2];
		const bool flag1 = v1[m-1]==v1[m-2];
		const bool flag2 = v2[m-1]==v2[m-2];
		//std::cout << m << " of " << std::min(n,max_idx) << std::endl;
		#ifdef ALGORITHM_PARALLEL
		tbb::parallel_for(std::max(0,m-n2), n1+1, [&] (int i1)
		#else
		for(int i1 = std::max(0,m-n2); i1 <= n1; i1++)
		#endif
		{
			const int i2_ub = std::min(max_idx-i1,n2);
			#ifdef ALGORITHM_PARALLEL
			tbb::parallel_for(std::max(0,m-i1), i2_ub+1, [&] (int i2)
			#else
			for(int i2 = std::max(0,m-i1); i2 <= i2_ub; i2++)
			#endif
			{
				//Rcpp::checkUserInterrupt();
				Q_new[i1][i2] = 0;
				const int k1_lb = flag1 ? i1 : std::max(0,m-1-i2);
				const int k2_lb_ = flag2 ? i2 : std::max(0,m-1-i1);
				T val = 1;
				#ifdef ALGORITHM_PARALLEL
				std::vector<T>& coeffs1 = coeffs1_.local();
				std::vector<T>& coeffs2 = coeffs2_.local();
					#ifdef LOGN_SUMMATION
					std::vector<T>& sum = sum_.local();
					#endif
				#endif
				coeffs1[i1] = 1;
				for(int k1 = i1-1; k1 >= k1_lb; k1--) {
					val *= k1+1;
					val *= d1;
					val /= i1-k1;
					coeffs1[k1] = val;
				}
				val = 1;
				coeffs2[i2] = 1;
				for(int k2 = i2-1; k2 >= k2_lb_; k2--) {
					val *= k2+1;
					val *= d2;
					val /= i2-k2;
					coeffs2[k2] = val;
				}
				#ifdef LOGN_SUMMATION
				//SumHelper<T> s;
				sum.resize(0);
				#endif
				for(int k1 = k1_lb; k1 <= i1; k1++)
				{
					const int k2_lb = flag2 ? i2 : std::max(0,m-1-k1);
					for(int k2 = k2_lb; k2 <= i2; k2++) {
						#ifdef LOGN_SUMMATION
						//s.add(Q[k1][k2] * coeffs1[k1] * coeffs2[k2]);
						sum.push_back(Q[k1][k2] * coeffs1[k1] * coeffs2[k2]);
						#else
						Q_new[i1][i2] += Q[k1][k2] * coeffs1[k1] * coeffs2[k2];
						#endif
					}
				}
				#ifdef LOGN_SUMMATION
				//Q_new[i1][i2] = s.sum();
				Q_new[i1][i2] = binary_sum(sum);
				#endif
				if(i1+i2 == m) res[i1][i2] = Q_new[i1][i2];
			}
			#ifdef ALGORITHM_PARALLEL
			);
			#endif
		}
		#ifdef ALGORITHM_PARALLEL
		);
		#endif
		std::swap(Q,Q_new);
	}
	return res;
}