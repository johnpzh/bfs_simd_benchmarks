#ifndef PEG_COUNT_H
#define PEG_COUNT_H
#include <stdio.h>
#include <float.h>
#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>
#include "peg_util.h"

using std::vector;
using std::map;
using std::pair;
////////////////////////////////////////////////////////////
// Count the access times
class AccessCounter {
private:
	unsigned long *counts = nullptr;
	unsigned long nnodes = 0;
	using pair_type = pair<unsigned long, unsigned long>;

	//struct pair_cmp {
	//	bool operator() (pair_type p1, pair_type p2) {
	//		return p1.second > p2.second;
	//	}
	//};

public:
	void init(unsigned long n) 
	{
		nnodes = n;
		free(counts);
		counts = (unsigned long *) calloc(n, sizeof(unsigned long));
	}

	void count(unsigned long node) 
	{
		++counts[node];
	}

	void atomic_count(unsigned long node) 
	{
		volatile unsigned long old_val;
		volatile unsigned long new_val;
		do {
			old_val = counts[node];
			new_val = counts[node] + 1;
		} while (!peg_CAS(counts + node, old_val, new_val));
	}

	void print()
	{
		//// Copy values into a map
		//map<unsigned long, unsigned long> counter;
		//for (unsigned long i = 0; i < nnodes; ++i) {
		//	counter[i] = counts[i];
		//}

		//// Sort the map
		//auto pair_cmp = [&] (pair_type p1, pair_type p2) -> bool {
		//	return p1.second > p2.second;
		//};
		//std::sort(counter.begin(), counter.end(), pair_cmp);

		//// Output
		//for (auto p_i = counter.begin(); p_i != counter.end(); ++p_i) {
		//	printf("%lu: %lu\n", p_i->first, p_i->second);
		//}

		std::sort(counts, counts + nnodes, std::greater<unsigned long>());
		for (unsigned long i = 0; i < nnodes; ++i) {
			printf("%lu %lu\n", i, counts[i]);
		}
	}

	virtual ~AccessCounter() 
	{
		free(counts);
	}
};
static AccessCounter bot_access_counter;
// End Count the access times
////////////////////////////////////////////////////////////

#endif

