#ifndef PEG_H
#define PEG_H
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
#endif
