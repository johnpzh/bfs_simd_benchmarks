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

//Structure to hold a node information
struct Node
{
	int starting;
	int num_of_edges;
};

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


void BFS_kernel(
		Node *h_graph_nodes,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		int *h_graph_visited,
		unsigned *graph_edges,
		int *h_cost)
{

	double start_time = omp_get_wtime();
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;

		omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for schedule(dynamic, 512)
		for(unsigned int tid = 0; tid < NNODES; tid++ )
		{
			if (h_graph_mask[tid] == true) {
				h_graph_mask[tid]=false;
				int next_starting = h_graph_nodes[tid].starting + h_graph_nodes[tid].num_of_edges;
				for(int i = h_graph_nodes[tid].starting; 
						i < next_starting; 
						i++)
				{
					int id = graph_edges[i];
					if(!h_graph_visited[id])
					{
						h_cost[id]=h_cost[tid]+1;
						h_updating_graph_mask[id]=true;
					}
				}
			}
		}

#pragma omp parallel for schedule(dynamic, 512)
		for(unsigned int tid=0; tid< NNODES ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true) {
				h_graph_mask[tid]=true;
				h_graph_visited[tid]=true;
				stop = false;
				h_updating_graph_mask[tid]=false;
			}
		}
	}
	while(!stop);
	double end_time = omp_get_wtime();
	printf("%d %lf\n", NUM_THREADS, (end_time - start_time));
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

	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*NNODES);
	int *h_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_graph_visited = (int*) malloc(sizeof(int)*NNODES);
	int* h_cost = (int*) malloc(sizeof(int)*NNODES);
	unsigned source = 0;
	unsigned edge_start = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		h_graph_nodes[i].starting = edge_start;
		h_graph_nodes[i].num_of_edges = graph_degrees[i];
		edge_start += graph_degrees[i];
	}

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("Input finished: %s\n", filename);
	unsigned run_count = 9;
#else
	unsigned run_count = 9;
#endif
	// BFS
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#ifndef ONEDEBUG
		//sleep(10);
#endif
		// Re-initializing
		memset(h_graph_mask, 0, sizeof(int)*NNODES);
		h_graph_mask[source] = 1;
		memset(h_updating_graph_mask, 0, sizeof(int)*NNODES);
		memset(h_graph_visited, 0, sizeof(int)*NNODES);
		h_graph_visited[source] = 1;
		for (unsigned i = 0; i < NNODES; ++i) {
			h_cost[i] = -1;
		}
		h_cost[source] = 0;

		BFS_kernel(
				h_graph_nodes,
				h_graph_mask,
				h_updating_graph_mask,
				h_graph_visited,
				graph_edges,
				h_cost);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);

	//Store the result into a file

#ifdef ONEDEBUG
	NUM_THREADS = 64;
	omp_set_num_threads(NUM_THREADS);
	unsigned num_lines = NNODES / NUM_THREADS;
#pragma omp parallel
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * num_lines;
	string file_prefix = "path/path";
	string file_name = file_prefix + to_string(tid) + ".txt";
	FILE *fpo = fopen(file_name.c_str(), "w");
	if (!fpo) {
		fprintf(stderr, "Error: connot open file %s.\n", file_name.c_str());
		exit(1);
	}
	unsigned bound_i;
	if (NUM_THREADS - 1 != tid) {
		bound_i = num_lines + offset;
	} else {
		bound_i = NNODES;
	}
	for (unsigned index = offset; index < bound_i; ++index) {
		fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
	}
	fclose(fpo);
}
#endif

	// Free memory
	free(graph_heads);
	free(graph_tails);
	free(graph_vertices);
	free(graph_edges);
	free(graph_degrees);
	free(h_graph_nodes);
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(h_graph_visited);
	free(h_cost);

	return 0;
}
