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
#include <algorithm>
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;
using std::vector;

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
		unsigned *&graph_vertices,
		unsigned *&graph_edges,
		unsigned *&graph_degrees)
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

	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_tails = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_vertices = (unsigned *) malloc(NNODES * sizeof(unsigned));
	graph_edges = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) malloc(NNODES * sizeof(unsigned));

	// Read degrees
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	for (unsigned i = 0; i < NNODES; ++i) {
		fscanf(fin, "%u", graph_degrees + i);
	}
	fclose(fin);

	NUM_THREADS = 64;
	unsigned edge_bound = NEDGES / NUM_THREADS;
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
		fscanf(fin, "%u %u", &NNODES, &NEDGES);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_tails[index] = n2;
		graph_edges[index] = n2;
	}

	fclose(fin);
}
	// CSR
	unsigned edge_start = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		graph_vertices[i] = edge_start;
		edge_start += graph_degrees[i];
	}
}

void print(unsigned *graph_cores) {
	FILE *foutput = fopen("ranks.txt", "w");
	unsigned kc = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		fprintf(foutput, "%u: %u\n", i+1, graph_cores[i]);
		if (kc < graph_cores[i]) {
			kc = graph_cores[i];
		}
	}
	fprintf(foutput, "kc: %u, KCORE: %u\n", kc, K_CORE);
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
	unsigned *graph_tails;
	unsigned *graph_vertices;
	unsigned *graph_edges;
	unsigned *graph_degrees;
	input(
		filename, 
		graph_heads, 
		graph_tails, 
		graph_vertices,
		graph_edges,
		graph_degrees);

	int *h_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_cost = (int*) malloc(sizeof(int)*NNODES);
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned *h_graph_parents = (unsigned *) malloc(sizeof(unsigned) * NNODES);
	unsigned source = 0;

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("Input finished: %s\n", input_f);
	unsigned run_count = 9;
#else
	unsigned run_count = 9;
#endif
	// BFS
	SIZE_BUFFER_MAX = 512;
	T_RATIO = 100;
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#ifndef ONEDEBUG
		//sleep(10);
#endif
		// Re-initializing
		for (unsigned k = 0; k < 3; ++k) {
		memset(h_graph_mask, 0, sizeof(int)*NNODES);
		memset(h_updating_graph_mask, 0, sizeof(int)*NNODES);
#pragma omp parallel for num_threads(64)
		for (unsigned j = 0; j < NNODES; ++j) {
			h_cost[j] = -1;
		}
		h_cost[source] = 0;
		memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
		memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);
#pragma omp parallel for num_threads(64)
		for (unsigned j = 0; j < NNODES; ++j) {
			h_graph_parents[j] = (unsigned) -1; // means unvisited yet
		}
		h_graph_parents[source] = source;

		graph_prepare(
				h_graph_vertices,
				h_graph_edges,
				h_graph_heads,
				h_graph_tails, 
				h_graph_degrees,
				h_graph_parents, // h_graph_visited
				tile_offsets,
				tile_sizes,
				source,
				h_graph_mask,
				h_updating_graph_mask,
				h_cost,
				is_active_side,
				is_updating_active_side);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
#ifdef ONEDEBUG
		printf("Thread %u finished.\n", NUM_THREADS);
#endif
		}
	}
	fclose(time_out);
#ifdef ONEDEBUG
	print(graph_cores);
#endif

	// Free memory
	free(graph_heads);
	free(graph_tails);
	free(graph_vertices);
	free(graph_edges);
	free(graph_degrees);
	free(graph_degrees_bak);
	free(graph_remain_mask);
	free(graph_updating_active);
	free(graph_cores);

	return 0;
}
