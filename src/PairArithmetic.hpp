#ifndef PAIR_ARITHMETIC_HPP
#define PAIR_ARITHMETIC_HPP

#include <limits>
#include <cmath>
#include <vector>
#include <forward_list>

#ifndef NO_K
#include<algorithm> //for std::max
#endif

#define within_range(T,T1) \
 ((std::numeric_limits<T>::digits <= std::numeric_limits<T1>::digits) && \
 (std::numeric_limits<T>::min_exponent >= std::numeric_limits<T1>::min_exponent) && \
 (std::numeric_limits<T>::max_exponent <= std::numeric_limits<T1>::max_exponent))
#define assert_within_range(T,T1) \
 static_assert(within_range(T,T1), \
  "Type cannot be exactly represented");

namespace PairArithmetic {
	template<typename T>
	T pow( T const& x, const long long exp ) {
		T ret(1);
		for(int i = 0; i < exp; i++) {
			ret *= x;
		}
		return ret;
	}

	template<typename T>
	T fast_pow( T base,unsigned long long exp ) {
		T result = T(1);
		while(exp) {
			if(exp & 1) result *= base;
			exp >>= 1;
			base *= base;
		}
		return result;
	}
	
	template<typename T>
	T factorial(const unsigned int n) {
		if(n==0) return 1;
		T ret = n;
		for(int i=1; i<n; i++) ret *= i;
		return ret;
	}
	
	template<typename T>
	T choose(const int n, const int k) {
		T res = 1;
		for(int i = 1; i <= k; i++) {
			res *= (n+1-i);
			res /= i;
		}
		return res;
	}
	
	template<typename T>
	class Pair {
		static_assert(std::numeric_limits<T>::is_iec559, "Type is not a recognized floating point type");
		private:
	#ifndef NO_K
			int k = 0;
	#endif
	#ifndef NO_OP
			T fl1, fl2;
			Pair(const T v1, const T v2) { fl1 = v1; fl2 = v2; }
	#else
			Pair(const T v1, const T v2) { }
	#endif
		public:
			Pair() : Pair(0,0) {}
	#ifndef NO_OP
			template<typename T1>
			Pair(const T1 val) : Pair((T)val, 0) { assert_within_range(T1,T); }
	#else
			template<typename T1>
			Pair(const T1 val) { assert_within_range(T1,T); }
	#endif
			template<typename T1> Pair<T>& operator+=( Pair<T1> const& y );
			template<typename T1> Pair<T>& operator-=( Pair<T1> const& y );
			template<typename T1> Pair<T>& operator*=( Pair<T1> const& y );
			template<typename T1> Pair<T>& operator/=( Pair<T1> const& y );
			template<typename T1> Pair<T>& operator+=( T1 const& y );
			template<typename T1> Pair<T>& operator-=( T1 const& y );
			template<typename T1> Pair<T>& operator*=( T1 const& y );
			template<typename T1> Pair<T>& operator/=( T1 const& y );
			template<typename T1> bool operator==(const Pair<T1>& other) const;
	#ifndef NO_OP
			explicit operator T() const { return this->fl1+this->fl2; }
			T getFl1() const { return this->fl1; }
			T getFl2() const { return this->fl2; }
	#endif
	#ifndef NO_K
			explicit operator T() const { return 0; }
			int getK() const {
				return k;
			}
			static int maxK() {
				return fast_pow(2,(std::numeric_limits<T>::digits-1)/2);
			}
	#endif
	};

	template<typename T> template<typename T1>
	inline bool Pair<T>::operator==(const Pair<T1>& other) const {
#ifndef NO_OP
	  return this->fl1 == other.fl1 && this->fl2 == other.fl2;
#else
	  return false;
#endif
}

	template<typename T>
	const T Fast2Sum( T& a, const T b ) {
		static_assert(std::numeric_limits<T>::is_iec559, "Type is not a recognized floating point type");
		const T s = a+b;
		const T z = s-a;
		a = s;
		return b-z;
	}

	template<typename T> template<typename T1>
	Pair<T>& Pair<T>::operator+=( Pair<T1> const& y ) {
		assert_within_range(T1,T);
	#ifndef NO_OP
		this->fl2 += Fast2Sum(this->fl1, (T)y.fl1);
		this->fl2 += y.fl2;
	#endif
	#ifndef NO_K
		this->k = std::max(this->k,y.k)+1;
	#endif
		return *this;
	}

	template<typename T> template<typename T1>
	Pair<T>& Pair<T>::operator-=( Pair<T1> const& y ) {
		assert_within_range(T1,T);
	#ifndef NO_OP
		this->fl2 += Fast2Sum(this->fl1, (T)-y.fl1);
		this->fl2 -= y.fl2;
	#endif
		return *this;
	}

	template<typename T> template<typename T1>
	Pair<T>& Pair<T>::operator*=( Pair<T1> const& y ) {
		assert_within_range(T1,T);
	#ifndef NO_OP
		const T c = this->fl1*y.fl1;
		this->fl2 *= y.fl1;
		this->fl2 += this->fl1 * y.fl2;
		this->fl2 += std::fma(this->fl1,(T)y.fl1,-c);
		this->fl1 = c;
	#endif
	#ifndef NO_K
		this->k += y.k + 1;
	#endif
		return *this;
	}

	template<typename T> template<typename T1>
	Pair<T>& Pair<T>::operator/=( Pair<T1> const& y ) {
		assert_within_range(T1,T);
	#ifndef NO_OP
		const T c = this->fl1 / y.fl1;
		const T t = std::fma((T)-y.fl1,c,this->fl1);
		this->fl2 = ((t+this->fl2) - c * y.fl2) / (y.fl1+y.fl2);
		this->fl1 = c;
	#endif
	#ifndef NO_K
		this->k += y.k + 2;
	#endif
		return *this;
	}

	template<typename T> template<typename T1>
	Pair<T>& Pair<T>::operator+=( T1 const& y ) {
		assert_within_range(T1,T);
	#ifndef NO_OP
		this->fl2 += Fast2Sum(this->fl1, (T)y);
	#endif
	#ifndef NO_K
		this->k = this->k+1;
	#endif
		return *this;
	}

	template<typename T> template<typename T1>
	Pair<T>& Pair<T>::operator-=( T1 const& y ) {
		assert_within_range(T1,T);
	#ifndef NO_OP
		this->fl2 += Fast2Sum(this->fl1, (T)-y);
	#endif
		return *this;
	}

	template<typename T> template<typename T1>
	Pair<T>& Pair<T>::operator*=( T1 const& y ) {
		assert_within_range(T1,T);
	#ifndef NO_OP
		const T c = this->fl1*y;
		this->fl2 *= (T)y;
		this->fl2 += std::fma(this->fl1,(T)y,-c);
		this->fl1 = c;
	#endif
	#ifndef NO_K
		this->k += 1;
	#endif
		return *this;
	}

	template<typename T> template<typename T1>
	Pair<T>& Pair<T>::operator/=( T1 const& y ) {
		assert_within_range(T1,T);
	#ifndef NO_OP
		const T c = this->fl1 / y;
		const T t = std::fma((T)-y,c,this->fl1);
		this->fl2 = (t+this->fl2) / y;
		this->fl1 = c;
	#endif
	#ifndef NO_K
		this->k += 2;
	#endif
		return *this;
	}
	
	template<typename T, typename T1> inline Pair<T> operator+ ( Pair<T> const& x, 	Pair<T1> const& y 	) { return Pair<T>(x) += y; }
	template<typename T, typename T1> inline Pair<T> operator+ ( Pair<T> const& x, 	T1 const& y 		) { return Pair<T>(x) += y; }
	template<typename T, typename T1> inline Pair<T> operator+ ( T1 const& y, 		Pair<T> const& x 	) { return Pair<T>(x) += y; }
	template<typename T, typename T1> inline Pair<T>&& operator+ ( Pair<T>&& x, T1 const& y ) { return static_cast<Pair<T>&&>(x += y); }
	
	template<typename T, typename T1> inline Pair<T> operator- ( Pair<T> const& x, 	Pair<T1> const& y 	) { return Pair<T>(x) -= y; }
	template<typename T, typename T1> inline Pair<T> operator- ( Pair<T> const& x, 	T1 const& y 		) { return Pair<T>(x) -= y; }
	template<typename T, typename T1> inline Pair<T> operator- ( T1 const& y, 		Pair<T> const& x 	) { return Pair<T>(y) -= x; }
	template<typename T, typename T1> inline Pair<T>&& operator- ( Pair<T>&& x, T1 const& y ) { return static_cast<Pair<T>&&>(x -= y); }
	
	template<typename T, typename T1> inline Pair<T> operator* ( Pair<T> const& x, 	Pair<T1> const& y 	) { return Pair<T>(x) *= y; }
	template<typename T, typename T1> inline Pair<T> operator* ( Pair<T> const& x, 	T1 const& y 		) { return Pair<T>(x) *= y; }
	template<typename T, typename T1> inline Pair<T> operator* ( T1 const& y, 		Pair<T> const& x 	) { return Pair<T>(x) *= y; }
	template<typename T, typename T1> inline Pair<T>&& operator* ( Pair<T>&& x, T1 const& y ) { return static_cast<Pair<T>&&>(x *= y); }
	
	template<typename T, typename T1> inline Pair<T> operator/ ( Pair<T> const& x, 	Pair<T1> const& y 	) { return Pair<T>(x) /= y; }
	template<typename T, typename T1> inline Pair<T> operator/ ( Pair<T> const& x, 	T1 const& y 		) { return Pair<T>(x) /= y; }
	template<typename T, typename T1> inline Pair<T> operator/ ( T1 const& y, 		Pair<T> const& x 	) { return Pair<T>(y) /= x; }
	template<typename T, typename T1> inline Pair<T>&& operator/ ( Pair<T>&& x, T1 const& y ) { return static_cast<Pair<T>&&>(x /= y); }


	template<typename T>
	class SumHelper {
		public:
			void add(const T& val) {
				if(!do_add) l.emplace_front(val);
				else l.front() += val;
				do_add = !do_add;
			}
			T sum() {
				while((++l.begin()) != l.end()) {
					typename std::forward_list<T>::iterator curr = l.begin();
					while(curr != l.end()) {
						typename std::forward_list<T>::iterator next = std::next(curr);
						if(next != l.end()) {
							*curr += *(next);
							curr = l.erase_after(curr);
						} else break;
					}
				}
				T ret = l.front();
				l.clear();
				do_add = false;
				return ret;
			}
		private:
			bool do_add = false;
			std::forward_list<T> l;
	};

	template<typename T>
	T binary_sum(std::vector<T> vec) {
		while(vec.size() > 1) {
			for(size_t i=0; i < vec.size()-1; i++){
				vec[i] += vec.back();
				vec.pop_back();
			}
		}
		if(vec.empty()) return T(0);
		else return(vec.back());
	}
	
	typedef Pair<double> DoublePair;
}
#endif