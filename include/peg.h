#ifndef PEG_H
#define PEG_H
#include <papi.h>
#include <vector>
#include <float.h>

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
	void print()
	{
		printf("cache access: %lld, cache misses: %lld, miss rate: %.2f%%\n", values[0], values[1], 100.0* values[1]/values[0]);
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
// End For PAPI
/////////////////////////////////////////////////////



/////////////////////////////////////////////////////
// For the Minimum Running time
unsigned NUM_THREADS_MIN;
unsigned RUNNING_TIME_MIN = LDBL_MAX;
void record_best_performance(double rt, unsigned num_thd)
{
	if (rt < runnindg_time_min) {
		RUNNING_TIME_MIN = rt;
		NUM_THREADS_MIN = num_thd;
	}
}

void print_best_performance()
{
	printf("Best_Performance:\n");
	printf("%u %f\n", NUM_THREADS_MIN, RUNNING_TIME_MIN);
}
// End For the Minimum Running time
/////////////////////////////////////////////////////
#endif
