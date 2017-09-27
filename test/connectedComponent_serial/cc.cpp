#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <unistd.h>
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
using std::to_string;

#define DUMP 0.85

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


//void input(char filename[], unsigned *&graph_heads, unsigned *&graph_ends, unsigned *&nneibor) 
void input(char filename[], unsigned *&graph_heads, unsigned *&graph_ends) 
{
	//printf("data: %s\n", filename);
	string prefix = string(filename) + "_untiled";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	fclose(fin);
	graph_heads = (unsigned *) malloc(2 * NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(2 * NEDGES * sizeof(unsigned));
	NUM_THREADS = 64;
	unsigned edge_bound = 2 * NEDGES / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * edge_bound;
	fname = prefix + "-" + to_string(tid);
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
		exit(1);
	}
	if (0 == tid) {
		fscanf(fin, "%u %u\n", &NNODES, &NEDGES);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = 2 * NEDGES;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_ends[index] = n2;
		++index;
		graph_heads[index] = n2;
		graph_ends[index] = n1;
	}

	fclose(fin);
}
}

void input_serial(char filename[], unsigned *&graph_heads, unsigned *&graph_ends)
{
	//printf("data: %s\n", filename);
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", filename);
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	graph_heads = (unsigned *) malloc(2 * NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(2 * NEDGES * sizeof(unsigned));
	for (unsigned i = 0; i < 2 * NEDGES; ++i) {
		unsigned head;
		unsigned end;
		fscanf(fin, "%u %u", &head, &end);
		--head;
		--end;
		graph_heads[i] = head;
		graph_ends[i] = end;
		++i;
		graph_heads[i] = end;
		graph_ends[i] = head;
	}
	fclose(fin);
}

void print(int cc_count) {
	printf("Conneted Component: %u\n", cc_count);
}

void sssp_kernel(
				unsigned *graph_heads, 
				unsigned *graph_ends, 
				int *graph_active, 
				int *graph_updating_active, 
				unsigned *graph_component,
				unsigned edge_i_start, 
				unsigned edge_i_bound)
{
#pragma omp parallel for
	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		unsigned end = graph_ends[edge_i];
		if (0 == graph_active[end]) {
			continue;
		}
		if (graph_component[head] > graph_component[end]) {
			graph_updating_active[head] = 1;
			graph_component[head] = graph_component[end];
		}
	}
}
//void page_rank(unsigned *graph_heads, unsigned *graph_ends, unsigned *nneibor, float *rank, float *sum) {}
void sssp(
		unsigned *graph_heads, 
		unsigned *graph_ends, 
		int *graph_active, 
		int *graph_updating_active,
		unsigned *graph_component)
{
	unsigned visited_count = 1;
	//for(int i=0;i<10;i++) {
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	int stop = 0;
	while (!stop) {
		stop = 1;
		sssp_kernel(
				graph_heads, 
				graph_ends, 
				graph_active, 
				graph_updating_active, 
				graph_component,
				0, 
				2 * NEDGES);
#pragma omp parallel for
		for (unsigned i = 0; i < NNODES; ++i) {
			if (1 == graph_updating_active[i]) {
				graph_updating_active[i] = 0;
				graph_active[i] = 1;
				stop = 0;
				++visited_count;
			} else {
				graph_active[i] = 0;
			}
		}

	}

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
}


int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 1) {
		filename = argv[1];
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_ends;
	//unsigned *nneibor;
#ifdef ONESERIAL
	//input_serial("/home/zpeng/benchmarks/data/fake/data.txt", graph_heads, graph_ends);
	input_serial("/home/zpeng/benchmarks/data/fake/mun_twitter", graph_heads, graph_ends);
#else
	input(filename, graph_heads, graph_ends);
#endif

	// Connected Component
	int *graph_active = (int *) malloc(NNODES * sizeof(int));
	int *graph_updating_active = (int *) malloc(NNODES * sizeof(int));
	unsigned *graph_component = (unsigned *) malloc(NNODES * sizeof(unsigned));
	
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned run_count = 2;
	printf("Start cc...\n");
#else
	unsigned run_count = 9;
#endif
	unsigned source = 0;
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		memset(graph_active, 0, NNODES * sizeof(int));
		for (unsigned k = 0; k < NNODES; ++k) {
			graph_active[k] = 1;
		}
		for (unsigned k = 0; k < NNODES; ++k) {
			graph_component[k] = k;
		}
		//sleep(10);
		sssp(
			graph_heads, 
			graph_ends, 
			graph_active, 
			graph_updating_active, 
			graph_component);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);
#ifdef ONEDEBUG
	unsigned cc_count = NNODES;
	for (unsigned i = 0; i < NNODES; ++i) {
		if (cc_count > graph_component[i]) {
			cc_count = graph_component[i];
		}
	}
	print(cc_count + 1);
#endif

	// Free memory
	free(graph_heads);
	free(graph_ends);
	free(graph_active);
	free(graph_updating_active);

	return 0;
}
