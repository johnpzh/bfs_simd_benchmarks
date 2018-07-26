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

int nnodes, nedges;
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
	fscanf(fin, "%u %u", &nnodes, &nedges);
	fclose(fin);
	graph_heads = (unsigned *) malloc(nedges * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(nedges * sizeof(unsigned));
	NUM_THREADS = 64;
	unsigned edge_bound = nedges / NUM_THREADS;
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
		fscanf(fin, "%u %u\n", &nnodes, &nedges);
	}
	if (NUM_THREADS - 1 != tid) {
		for (unsigned i = 0; i < edge_bound; ++i) {
			unsigned index = i + offset;
			unsigned n1;
			unsigned n2;
			fscanf(fin, "%u %u", &n1, &n2);
			n1--;
			n2--;
			graph_heads[index] = n1;
			graph_ends[index] = n2;
		}
	} else {
		for (unsigned i = 0; i + offset < nedges; ++i) {
			unsigned index = i + offset;
			unsigned n1;
			unsigned n2;
			fscanf(fin, "%u %u", &n1, &n2);
			n1--;
			n2--;
			graph_heads[index] = n1;
			graph_ends[index] = n2;
		}
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
	fscanf(fin, "%u %u", &nnodes, &nedges);
	graph_heads = (unsigned *) malloc(nedges * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(nedges * sizeof(unsigned));
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned head;
		unsigned end;
		fscanf(fin, "%u %u", &head, &end);
		--head;
		--end;
		graph_heads[i] = head;
		graph_ends[i] = end;
	}
	fclose(fin);
}

void print(int *dists) {
	FILE *fout = fopen("distances.txt", "w");
	for(unsigned i=0;i<nnodes;i++) {
		fprintf(fout, "%d\n", dists[i]);
	}
	fclose(fout);
}

void sssp_kernel(unsigned *graph_heads, unsigned *graph_ends, int *graph_active, int *graph_updating_active,int *dists, unsigned edge_i_start, unsigned edge_i_bound)
{
#pragma omp parallel for
	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		if (0 == graph_active[head]) {
			continue;
		}
		unsigned end = graph_ends[edge_i];
		if (-1 == dists[end] || dists[head] + 1 < dists[end]) {
			dists[end] = dists[head] + 1;
			graph_updating_active[end] = 1;
		}
	}
}
//void page_rank(unsigned *graph_heads, unsigned *graph_ends, unsigned *nneibor, float *rank, float *sum) {}
void sssp(unsigned *graph_heads, unsigned *graph_ends, int *graph_active, int *graph_updating_active,int *dists)
{
	//for(int i=0;i<10;i++) {
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	int stop = 0;
	while (!stop) {
		stop = 1;
		sssp_kernel(graph_heads, graph_ends, graph_active, graph_updating_active, dists, 0, nedges);
#pragma omp parallel for
		for (unsigned i = 0; i < nnodes; ++i) {
			if (1 == graph_updating_active[i]) {
				graph_updating_active[i] = 0;
				graph_active[i] = 1;
				stop = 0;
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
	input_serial("data.txt", graph_heads, graph_ends);
#else
	input(filename, graph_heads, graph_ends);
#endif

	// PageRank
	//float *rank = (float *) malloc(nnodes * sizeof(float));
	//float *sum = (float *) malloc(nnodes * sizeof(float));
	int *distances = (int *) malloc(nnodes * sizeof(int));
	int *graph_active = (int *) malloc(nnodes * sizeof(int));
	int *graph_updating_active = (int *) malloc(nnodes * sizeof(int));
	
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned run_count = 2;
#else
	unsigned run_count = 9;
#endif
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		memset(distances, -1, nnodes * sizeof(int));
		distances[0] = 0;
		memset(graph_active, 0, nnodes * sizeof(int));
		memset(graph_updating_active, 0, nnodes * sizeof(int));
		graph_active[0] = 1;
		//sleep(10);
		//page_rank(graph_heads, graph_ends, nneibor, rank, sum);
		sssp(graph_heads, graph_ends, graph_active, graph_updating_active, distances);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);
#ifdef ONEDEBUG
	print(distances);
#endif

	// Free memory
	//free(nneibor);
	free(graph_heads);
	free(graph_ends);
	//free(rank);
	//free(sum);
	free(distances);
	free(graph_active);
	free(graph_updating_active);

	return 0;
}
