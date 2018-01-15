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
unsigned K_CORE;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


//void input(char filename[], unsigned *&graph_heads, unsigned *&graph_tails, unsigned *&nneibor) 
void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
		unsigned *&graph_vertices,
		unsigned *&graph_edges,
		unsigned *&graph_degrees)
		//unsigned *&graph_adj_indices)
		//(vector<vector<unsigned>> &graph_neighbors) 
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
		//graph_vertices_info[i].out_neighbors = graph_edges + edge_start;
		//graph_vertices_info[i].out_neighbors = h_graph_edges + edge_start;
		//graph_vertices_info[i].out_degree = h_graph_degrees[i];
	}
	

}

void input_serial(
				char filename[], 
				unsigned *&graph_heads, 
				unsigned *&graph_tails, 
				unsigned *&graph_degrees,
				unsigned *&graph_adj_indices)
{
	//printf("data: %s\n", filename);
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", filename);
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_tails = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) calloc(NNODES, sizeof(unsigned));
	for (unsigned i = 0; i < NEDGES; ++i) {
		unsigned head;
		unsigned end;
		fscanf(fin, "%u %u", &head, &end);
		--head;
		--end;
		graph_heads[i] = head;
		graph_tails[i] = end;
		graph_degrees[head]++;
		//graph_degrees[end]++;
	}
	fclose(fin);

	graph_adj_indices = (unsigned *) malloc(NNODES * sizeof(unsigned));
	unsigned offset = 0;
	graph_adj_indices[0] = offset;
	for (unsigned i = 0; i < NNODES - 1; ++i) {
		offset += graph_degrees[i];
		graph_adj_indices[i + 1] = offset;
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
//void kcore_kernel(
//				unsigned *graph_heads, 
//				unsigned *graph_tails,
//				unsigned *graph_degrees,
//				unsigned *graph_adj_indices,
//				int *graph_updating_active, 
//				const unsigned &edge_i_start, 
//				const unsigned &edge_i_bound,
//				unsigned *graph_cores)
//{
////#pragma omp parallel for
//	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
//		//unsigned head = graph_heads[edge_i];
//		unsigned end = graph_tails[edge_i];
//		//if (graph_updating_active[head] && graph_degrees[end]) {}
//		if (graph_degrees[end]) {
////#pragma omp atomic
//			graph_degrees[end]--;
//			if (!graph_degrees[end]) {
//				graph_cores[end] = K_CORE - 1;
//			}
//		}
//	}
//}
inline void kcore_kernel(
		unsigned *graph_heads,
		unsigned *graph_tails,
		unsigned *graph_vertices,
		unsigned *graph_edges,
		unsigned *graph_degrees,
		unsigned *graph_degrees_bak,
		int *graph_updating_active,
		unsigned *graph_cores)
{
#pragma omp parallel for schedule(dynamic, 128)
	for (unsigned h_id = 0; h_id < NNODES; ++h_id) {
		if (!graph_updating_active[h_id]) {
			continue;
		}
		unsigned bound_edge_i = graph_vertices[h_id] + graph_degrees_bak[h_id];
		for (unsigned edge_i = graph_vertices[h_id]; edge_i < bound_edge_i; ++edge_i) {
			unsigned tail_id = graph_edges[edge_i];
			if (graph_degrees[tail_id] > 0) {
				//--graph_degrees[tail_id];
				volatile unsigned old_val = graph_degrees[tail_id];
				volatile unsigned new_val = old_val - 1;
				while (!__sync_bool_compare_and_swap(graph_degrees + tail_id, old_val, new_val)) {
					old_val = graph_degrees[tail_id];
					new_val = old_val - 1;
				}
			}
		}
		graph_updating_active[h_id] = 0;
	}
}
void kcore(
		unsigned *graph_heads, 
		unsigned *graph_tails, 
		unsigned *graph_vertices,
		unsigned *graph_edges,
		unsigned *graph_degrees,
		unsigned *graph_degrees_bak,
		//unsigned *graph_adj_indices,
		int *graph_remain_mask,
		int *graph_updating_active,
		unsigned *graph_cores)
{
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	int stop = 0;
	while (!stop) {
		stop = 1;
		K_CORE++;
		while (true) {
			bool has_remove = false;
#pragma omp parallel for schedule(dynamic, 128)
			for (unsigned i = 0; i < NNODES; ++i) {
				if (!graph_remain_mask[i]) {
					continue;
				}
				stop = 0;
				if (graph_degrees[i] < K_CORE) {
					graph_updating_active[i] = 1;
					graph_remain_mask[i] = 0;
					graph_degrees[i] = 0;
					graph_cores[i] = K_CORE - 1;
					has_remove = true;
				}
			}
			if (!has_remove) {
				break;
			}
			kcore_kernel(
					graph_heads,
					graph_tails,
					graph_vertices,
					graph_edges,
					graph_degrees,
					graph_degrees_bak,
					graph_updating_active,
					graph_cores);
		}
		printf("K_CORE: %u\n", K_CORE);//test
	}
	K_CORE -= 2;

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
		//filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		//filename = "/home/zpeng/benchmarks/data/skitter/out.skitter";
		filename = "/sciclone/scr-mlt/zpeng01/skitter/out.skitter";

	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_tails;
	unsigned *graph_vertices;
	unsigned *graph_edges;
	unsigned *graph_degrees;
#ifdef ONESERIAL
	//input_serial("/home/zpeng/benchmarks/data/fake/data.txt", graph_heads, graph_tails, graph_degrees);
	//input_serial("/home/zpeng/benchmarks/data/fake/mun_twitter", graph_heads, graph_tails,graph_degrees);
	input_serial(
				"/home/zpeng/benchmarks/data/zebra/out.zebra_sym", 
				graph_heads, 
				graph_tails,
				graph_degrees,
				graph_adj_indices);
#else
	input(
		filename, 
		graph_heads, 
		graph_tails, 
		graph_vertices,
		graph_edges,
		graph_degrees);
#endif

	// K-core
	int *graph_remain_mask = (int *) malloc(NNODES * sizeof(int));
	int *graph_updating_active = (int *) malloc(NNODES * sizeof(int));
	unsigned *graph_cores = (unsigned *) malloc(NNODES * sizeof(unsigned));
	unsigned *graph_degrees_bak = (unsigned *) malloc(NNODES * sizeof(unsigned));
	memcpy(graph_degrees_bak, graph_degrees, NNODES * sizeof(unsigned));
	
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned run_count = 9;
	printf("Start K-core...\n");
#else
	unsigned run_count = 9;
#endif
	for (unsigned i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		memset(graph_updating_active, 0, NNODES * sizeof(int));
		for (unsigned k = 0; k < NNODES; ++k) {
			graph_remain_mask[k] = 1;
			graph_cores[k] = 0;
		}
		K_CORE = 0;
		memcpy(graph_degrees, graph_degrees_bak, NNODES * sizeof(unsigned));
		//sleep(10);
		kcore(
			graph_heads, 
			graph_tails, 
			graph_vertices,
			graph_edges,
			graph_degrees,
			graph_degrees_bak,
			graph_remain_mask,
			graph_updating_active,
			graph_cores);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
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
