#ifndef PEG_PROFILING_H
#define PEG_PROFILING_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <float.h>
#include <string.h>
#include <papi.h>

////////////////////////////////////////////////
// For PAPI, cache miss rate
// PAPI test results
class CacheMissRate {
public:
	void measure_start()
	{
		int retval;
		if ((retval = PAPI_start_counters(events, 2)) < PAPI_OK) {
			test_fail(__FILE__, __LINE__, "PAPI_start_counters", retval);
		}
	}
	void measure_stop()
	{
		int retval;
		if ((retval = PAPI_stop_counters(values, 2)) < PAPI_OK) {
			test_fail(__FILE__, __LINE__, "PAPI_stop_counters", retval);
		}
	}
	void print(unsigned metrics = (unsigned) -1)
	{
		if (metrics == (unsigned) -1) {
			printf("cache access: %lld, cache misses: %lld, miss rate: %.2f%%\n", values[0], values[1], 100.0* values[1]/values[0]);
		} else {
			printf("%u %.2f\n", metrics, 1.0 * values[1]/values[0]);
		}
	}

private:
	int events[2] = { PAPI_L2_TCA, PAPI_L2_TCM};
	long long values[2];

	void test_fail(char *file, int line, char *call, int retval)
	{
		printf("%s\tFAILED\nLine # %d\n", file, line);
		if ( retval == PAPI_ESYS ) {
			char buf[128];
			memset( buf, '\0', sizeof(buf) );
			sprintf(buf, "System error in %s:", call );
			perror(buf);
		}
		else if ( retval > 0 ) {
			printf("Error calculating: %s\n", call );
		}
		else {
			printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
		}
		printf("\n");
		exit(1);
	}
};
static CacheMissRate bot_miss_rate;
// End For PAPI
/////////////////////////////////////////////////////



/////////////////////////////////////////////////////
// For the Minimum Running time
//unsigned NUM_THREADS_MIN;
//double RUNNING_TIME_MIN = DBL_MAX;
//void record_best_performance(double rt, unsigned num_thd)
//{
//	if (rt < RUNNING_TIME_MIN) {
//		RUNNING_TIME_MIN = rt;
//		NUM_THREADS_MIN = num_thd;
//	}
//}
//
//void print_best_performance()
//{
//	printf("Best_Performance:\n");
//	printf("%u %f\n", NUM_THREADS_MIN, RUNNING_TIME_MIN);
//}

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

	void print_best_performance()
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
	void CAS()

public:
	void record_simd(unsigned long eff, unsigned long all) 
	{
		effect += eff;
		total += all;
		if (eff > all) {
			printf("Crazy: eff: %u, all: %u\n", eff, all);
		}
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
