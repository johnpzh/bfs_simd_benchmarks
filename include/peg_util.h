#ifndef PEG_UTIL_H
#define PEG_UTIL_H
#include <stdio.h>
#include <float.h>

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
	unsigned NUM_THREADS_MIN;
	double RUNNING_TIME_MIN = DBL_MAX;

public:
	void record_best_performance(double rt, unsigned num_thd)
	{
		if (rt < RUNNING_TIME_MIN) {
			RUNNING_TIME_MIN = rt;
			NUM_THREADS_MIN = num_thd;
		}
	}

	void print()
	{
		printf("Best_Performance:\n");
		printf("%u %f\n", NUM_THREADS_MIN, RUNNING_TIME_MIN);
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
	void record_simd(unsigned long eff, unsigned long all) 
	{
		unsigned long old_val;
		unsigned long new_val;
		//TODO peg_CAS()
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
#endif
