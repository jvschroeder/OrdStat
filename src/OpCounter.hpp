#ifndef OP_COUNTER_HPP
#define OP_COUNTER_HPP
class OpCounter {
	public:
		static uint64_t global_k;
		template<typename T>
		OpCounter(T const&) {}
		OpCounter(){}
		template<typename T>
		explicit operator T() const { return 0; }
		OpCounter& operator+=( OpCounter const &y );
		OpCounter& operator-=( OpCounter const &y );
		OpCounter& operator*=( OpCounter const &y );
		OpCounter& operator/=( OpCounter const &y );
		bool operator==( OpCounter const &y ) const { return false; }
		static void reset() { global_k = 0; }
		static void inc() { ++global_k; }
} counter;

uint64_t OpCounter::global_k = 0;

OpCounter& OpCounter::operator+=( OpCounter const &y ) { OpCounter::inc(); return counter; }
OpCounter& OpCounter::operator-=( OpCounter const &y ) { OpCounter::inc(); return counter; }
OpCounter& OpCounter::operator*=( OpCounter const &y ) { OpCounter::inc(); return counter; }
OpCounter& OpCounter::operator/=( OpCounter const &y ) { OpCounter::inc(); return counter; }

OpCounter& operator+ ( OpCounter x, OpCounter const& y ) { OpCounter::inc(); return counter; }
OpCounter& operator- ( OpCounter x, OpCounter const& y ) { OpCounter::inc(); return counter; }
OpCounter& operator* ( OpCounter x, OpCounter const& y ) { OpCounter::inc(); return counter; }
OpCounter& operator/ ( OpCounter x, OpCounter const& y ) { OpCounter::inc(); return counter; }

template<typename T>
OpCounter& operator+ ( OpCounter x, T const& y ) { OpCounter::inc(); return counter; }
template<typename T>
OpCounter& operator- ( OpCounter x, T const& y ) { OpCounter::inc(); return counter; }
template<typename T>
OpCounter& operator* ( OpCounter x, T const& y ) { OpCounter::inc(); return counter; }
template<typename T>
OpCounter& operator/ ( OpCounter x, T const& y ) { OpCounter::inc(); return counter; }
#endif