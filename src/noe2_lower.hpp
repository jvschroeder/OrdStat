//#ifdef ALGORITHM_PARALLEL
//#include "tbb/enumerable_thread_specific.h"
//#include "tbb/parallel_for.h"
//#endif
//#include<vector>

{
	#ifdef ALGORITHM_PARALLEL
	tbb::enumerable_thread_specific< std::vector<T> > coeffs1_(std::vector<T>(n1+1,-1));
	tbb::enumerable_thread_specific< std::vector<T> > coeffs2_(std::vector<T>(n2+1,-1));
	#else
	std::vector<T> coeffs1(n1+1,-1);
	std::vector<T> coeffs2(n2+1,-1);
  #ifdef PROGRESS
	std::vector<uint64_t> local_counts(std::max(max_idx,n2)+1,0);
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
	#ifdef PROGRESS
	const uint64_t n2_2 = n2*n2;
	const uint64_t n2_3 = n2_2*n2;
	const uint64_t n1_2 = n1*n1;
	const uint64_t n1_3 = n1_2*n1;
	const uint64_t tot = 0.5*n2+1.5*n2_2+n2_3+(1.5+1.5*n2+4.25*n2_2+1.25*n2_3)*n1+(1.5+4.25*n2+3*n2_2+0.25*n2_3)*n1_2+(1+1.25*n2+0.25*n2_2)*n1_3;
	ProgressBar bar(Rcpp::Rcout,tot);
	uint64_t cnt = (n1+1)*(n2+2);
	std::mutex cnt_mutex;
	#endif
	for(int m = 2; m <= std::min(n,max_idx); m++) {
		const T d1 = v1[m-1]-v1[m-2];
		const T d2 = v2[m-1]-v2[m-2];
		#ifdef PROGRESS
		cnt += 2;
		#endif
		const bool flag1 = v1[m-1]==v1[m-2];
		const bool flag2 = v2[m-1]==v2[m-2];
		#ifdef ALGORITHM_PARALLEL
		tbb::parallel_for(std::max(0,m-n2), n1+1, [&] (int i1)
		#else
		for(int i1 = std::max(0,m-n2); i1 <= n1; i1++)
		#endif
		{
		  const int i2_lb = std::max(0,m-i1);
			const int i2_ub = std::min(max_idx-i1,n2);
      #ifdef PROGRESS
        #ifdef ALGORITHM_PARALLEL
			  std::vector<uint64_t> local_counts(i2_ub+1,0);
        #endif
      #endif
			#ifdef ALGORITHM_PARALLEL
			tbb::parallel_for(i2_lb, i2_ub+1, [&] (int i2)
			#else
			for(int i2 = i2_lb; i2 <= i2_ub; i2++)
			#endif
			{
				RcppThread::checkUserInterrupt();
				Q_new[i1][i2] = 0;
				const int k1_lb = flag1 ? i1 : std::max(0,m-1-i2);
				const int k2_lb_ = flag2 ? i2 : std::max(0,m-1-i1);
				T val = 1;
				#ifdef ALGORITHM_PARALLEL
				std::vector<T>& coeffs1 = coeffs1_.local();
				std::vector<T>& coeffs2 = coeffs2_.local();
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
				#ifdef PROGRESS
				local_counts[i2] = i1+i2-k1_lb-k2_lb_;
				#endif
				for(int k1 = k1_lb; k1 <= i1; k1++)
				{
					const int k2_lb = flag2 ? i2 : std::max(0,m-1-k1);
					#ifdef PROGRESS
				  local_counts[i2] += i2-k2_lb+1;
					#endif
					for(int k2 = k2_lb; k2 <= i2; k2++) {
						Q_new[i1][i2] += Q[k1][k2] * coeffs1[k1] * coeffs2[k2];
					}
				}
				if(i1+i2 == m){ res[i1][i2] = Q_new[i1][i2]; }
			}
			#ifdef ALGORITHM_PARALLEL
			);
			#endif
			#ifdef PROGRESS
      #ifndef ALGORITHM_PARALLEL
			const uint64_t local_count = std::accumulate(&local_counts[i2_lb], &local_counts[i2_ub]+1,0);
      #else
			const uint64_t local_count = tbb::parallel_reduce(
			  tbb::blocked_range<uint64_t*>(&local_counts[i2_lb], &local_counts[i2_ub]+1),
			  0,
			  [](const tbb::blocked_range<uint64_t*>& r, uint64_t value)->uint64_t {
			    return std::accumulate(r.begin(),r.end(),value);
			  },
			  std::plus<uint64_t>()
			);
      #endif
			{
			  std::lock_guard<std::mutex> guard(cnt_mutex);
        cnt += 3*local_count;
			}
      #endif
      #ifdef PROGRESS
      		bar.show(cnt);
      #endif
		}
		#ifdef ALGORITHM_PARALLEL
		);
		#endif
		std::swap(Q,Q_new);
	}
  #ifdef PROGRESS
  	bar.show(cnt,true);
  #endif
	return res;
}