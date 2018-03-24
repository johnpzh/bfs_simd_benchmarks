#ifndef PEG_PAPI_H
#define PEG_PAPI_H
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


#endif
