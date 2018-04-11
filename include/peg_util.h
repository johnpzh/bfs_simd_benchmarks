#ifndef PEG_UTIL_H
#define PEG_UTIL_H
#include <stdio.h>
#include <float.h>
#include <immintrin.h>

////////////////////////////////////////////////////////////
// Campare and Swap
template <typename V_T>
inline bool peg_CAS(V_T *ptr, V_T old_val, V_T new_val)
{
	if (1 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((char *)ptr, *((char *) &old_val), *((char *) &new_val));
	} else if (4 == sizeof(V_T)) {
		return __sync_bool_compare_and_swap((int *)ptr, *((int *) &old_val), *((int *) &new_val));
	} else if (8 == sizeof(V_T) && 8 == sizeof(long)) {
		return __sync_bool_compare_and_swap((long *)ptr, *((long *) &old_val), *((long *) &new_val));
	} else {
		printf("CAS cannot support the type.\n");
		exit(1);
	}
}
// End CAS
////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////
// For the Minimum Running time
class BestPerform {
private:
	unsigned num_threads_min;
	double running_time_min = DBL_MAX;
	unsigned count = 0;
	double total = 0;

public:
	void record(double rt, unsigned num_thd)
	{
		if (rt < running_time_min) {
			running_time_min = rt;
			num_threads_min = num_thd;
		}
		total += rt;
		++count;
	}

	void print_best()
	{
		printf("Best_Performance: %f\n", running_time_min);
	}

	void print_average(unsigned metrics = (unsigned) -1)
	{
		if (metrics == (unsigned) -1) {
			printf("Average: %f\n", total/count);
		} else {
			printf("Average: %u %f\n", metrics, total/count);
		}
	}

	void reset()
	{
		count = 0;
		total = 0;
	}
};
static BestPerform bot_best_perform;
// End For the Minimum Running time
/////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
// SIMD Utilization
class SIMDUtil {
private:
	unsigned long effect = 0;
	unsigned long total = 0;

public:
	void record(unsigned long eff, unsigned long all) 
	{
		unsigned long old_val;
		unsigned long new_val;
		do {
			old_val = effect;
			new_val = effect + eff;
		} while (!peg_CAS(&effect, old_val, new_val));

		do {
			old_val = total;
			new_val = total + all;
		} while (!peg_CAS(&total, old_val, new_val));

		////////////////////////////////////
		//effect += eff;
		//total += all;
	}
	void print(unsigned metrics = (unsigned) -1) 
	{
		printf("effect: %lu, total: %lu\n", effect, total);//test;
		if (metrics == (unsigned) -1) {
			printf("SIMD Utilization: %.2f%%\n", (double) effect/total * 100.0);
		} else {
			printf("%u %f\n", metrics, 1.0 * effect/total);
		}
	}
	void reset() 
	{
		effect = 0;
		total = 0;
	}
};
static SIMDUtil bot_simd_util;
// End SIMD Utilization
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// Necessary Access Ratio
class NecessaryAccess {
private:
	unsigned long effect = 0;
	unsigned long total = 0;

	unsigned mmask16_count_one(__mmask16 m)
	{
		__m512i ones = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), m, 1);
		return _mm512_reduce_add_epi32(ones);
	}

public:
	void record(__mmask16 active_mask, unsigned long all) 
	{
		unsigned long old_val;
		unsigned long new_val;
		//unsigned long eff = mmask16_count_one(active_mask);
		//do {
		//	old_val = effect;
		//	new_val = effect + eff;
		//} while (!peg_CAS(&effect, old_val, new_val));

		do {
			old_val = total;
			new_val = total + all;
		} while (!peg_CAS(&total, old_val, new_val));

		////////////////////////////////////
		//effect += eff;
		//total += all;
	}
	void print(unsigned metrics = (unsigned) -1) 
	{
		printf("effect: %lu, total: %lu\n", effect, total);//test;
		if (metrics == (unsigned) -1) {
			//printf("Necessary Access: %.2f%%\n", (double) effect/total * 100.0);
			printf("Total Access: %lu\n", total);
		} else {
			printf("%u %f\n", metrics, 1.0 * effect/total);
		}
	}
	void reset() 
	{
		effect = 0;
		total = 0;
	}
};
static NecessaryAccess bot_necessary_access;
// End Necessary Access Ratio
////////////////////////////////////////////////////////////
#endif
