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
#include "../../include/peg.h"
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
using std::to_string;
using std::vector;
using std::pair;
using std::map;


unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH; // Width of tile
unsigned SIDE_LENGTH; // Number of rows of tiles
unsigned NUM_TILES; // Number of tiles
unsigned ROW_STEP; // Number of rows of tiles in a Group

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_ends,
		unsigned *&tile_offsets,
		int *&is_empty_tile) 
{
	//printf("data: %s\n", filename);
	//string prefix = string(filename) + "_untiled";
	string prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	fclose(fin);
	if (NNODES % TILE_WIDTH) {
		SIDE_LENGTH = NNODES / TILE_WIDTH + 1;
	} else {
		SIDE_LENGTH = NNODES / TILE_WIDTH;
	}
	NUM_TILES = SIDE_LENGTH * SIDE_LENGTH;
	// Read tile Offsets
	fname = prefix + "-offsets";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	tile_offsets = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		//fscanf(fin, "%u", tile_offsets + i);
		unsigned offset;
		fscanf(fin, "%u", &offset);
		tile_offsets[i] = offset;
	}
	fclose(fin);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
	memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			if (tile_offsets[i] == tile_offsets[i + 1]) {
				is_empty_tile[i] = 1;
			}
		} else {
			if (tile_offsets[i] == NEDGES) {
				is_empty_tile[i] = 1;
			}
		}
	}
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
		fscanf(fin, "%u %u\n", &NNODES, &NEDGES);
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
		graph_ends[index] = n2;
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
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	for (unsigned i = 0; i < NEDGES; ++i) {
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

void print(unsigned *graph_component) 
{
	FILE *foutput = fopen("ranks.txt", "w");
	map<unsigned, unsigned> concom;
	for (unsigned i = 0; i < NNODES; ++i) {
		unsigned comp_id = graph_component[i];
		fprintf(foutput, "%u: %u\n", i, comp_id);
		if (concom.find(comp_id) == concom.end()) {
			concom.insert(pair<unsigned, unsigned>(comp_id, 0));
			concom[comp_id]++;
		} else {
			concom[comp_id]++;
		}
	}
	fprintf(foutput, "Number of CC: %lu\n", concom.size());
	unsigned lcc = 0;
	unsigned max_count = 0;
	for (auto it = concom.begin(); it != concom.end(); ++it) {
		if (max_count < it->second) {
			max_count = it->second;
			lcc = it->first;
		}
	}
	fprintf(foutput, "Size of LCC: %u\n", max_count);
	fprintf(foutput, "LCC ID: %u\n", lcc);
}

inline void cc_kernel(
				unsigned *graph_heads, 
				unsigned *graph_ends, 
				int *graph_active, 
				int *graph_updating_active, 
				int *is_active_side,
				int *is_updating_active_side,
				unsigned *graph_component,
				const unsigned &edge_i_start, 
				const unsigned &edge_i_bound)
{
//#pragma omp parallel for
	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		unsigned end = graph_ends[edge_i];
		if (1 == graph_active[head]) {
			if (graph_component[head] < graph_component[end]) {
				graph_updating_active[end] = 1;
				graph_component[end] = graph_component[head];
				is_updating_active_side[end/TILE_WIDTH] = 1;
			}
		}
		//if (1 == graph_active[end]) {
		//	if (graph_component[end] < graph_component[head]) {
		//		graph_updating_active[head] = 1;
		//		graph_component[head] = graph_component[end];
		//		is_updating_active_side[head/TILE_WIDTH] = 1;
		//	}
		//}
	}
}

inline void scheduler(
					unsigned *graph_heads, 
					unsigned *graph_ends, 
					unsigned *tile_offsets,
					int *graph_active, 
					int *graph_updating_active,
					int *is_active_side,
					int *is_updating_active_side,
					int *is_empty_tile,
					unsigned *graph_component,
					const unsigned &start_row_index,
					const unsigned &bound_row_index)
{
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned col_id = 0; col_id < SIDE_LENGTH; ++col_id) {
		for (unsigned row_id = start_row_index; row_id < bound_row_index; ++row_id) {
			unsigned tile_id = row_id * SIDE_LENGTH + col_id;
			if (is_empty_tile[tile_id]) {
				continue;
			}
			//bfs_kernel();
			unsigned bound_edge_i;
			if (NUM_TILES - 1 != tile_id) {
				bound_edge_i = tile_offsets[tile_id + 1];
			} else {
				bound_edge_i = NEDGES;
			}
			cc_kernel(
				graph_heads, 
				graph_ends, 
				graph_active, 
				graph_updating_active,
				is_active_side,
				is_updating_active_side,
				graph_component,
				tile_offsets[tile_id], 
				bound_edge_i);
		}
	}
}

void cc(
		unsigned *graph_heads, 
		unsigned *graph_ends, 
		unsigned *tile_offsets,
		int *graph_active, 
		int *graph_updating_active,
		int *is_active_side,
		int *is_updating_active_side,
		int *is_empty_tile,
		unsigned *graph_component)
{
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	int stop = 0;
	while (!stop) {
		stop = 1;
		unsigned side_id;
		for (side_id = 0; side_id + ROW_STEP <= SIDE_LENGTH; ) {
			if (!is_active_side[side_id]) {
				++side_id;
				continue;
			}
			scheduler(
				graph_heads, 
				graph_ends, 
				tile_offsets,
				graph_active, 
				graph_updating_active,
				is_active_side,
				is_updating_active_side,
				is_empty_tile,
				graph_component,
				side_id,
				side_id + ROW_STEP);
			side_id += ROW_STEP;
		}
		scheduler(
			graph_heads, 
			graph_ends, 
			tile_offsets,
			graph_active, 
			graph_updating_active,
			is_active_side,
			is_updating_active_side,
			is_empty_tile,
			graph_component,
			side_id,
			SIDE_LENGTH);
#pragma omp parallel for
		//for (unsigned i = 0; i < NNODES; ++i) {
		//	if (1 == graph_updating_active[i]) {
		//		graph_updating_active[i] = 0;
		//		graph_active[i] = 1;
		//		stop = 0;
		//	} else {
		//		graph_active[i] = 0;
		//	}
		//}
		for (unsigned side_id = 0; side_id < SIDE_LENGTH; ++side_id) {
			if (!is_updating_active_side[side_id]) {
				is_active_side[side_id] = 0;
				continue;
			}
			is_updating_active_side[side_id] = 0;
			is_active_side[side_id] = 1;
			stop = 0;
			unsigned bound_vertex_id;
			if (SIDE_LENGTH - 1 != side_id) {
				bound_vertex_id = side_id * TILE_WIDTH + TILE_WIDTH;
			} else {
				bound_vertex_id = NNODES;
			}
			for (unsigned vertex_id = side_id * TILE_WIDTH; vertex_id < bound_vertex_id; ++vertex_id) {
				if (1 == graph_updating_active[vertex_id]) {
					graph_updating_active[vertex_id] = 0;
					graph_active[vertex_id] = 1;
				} else {
					graph_active[vertex_id] = 0;
				}
			}
		}
	}

	double end_time = omp_get_wtime();
	double rt;
	printf("%u %lf\n", NUM_THREADS, rt = end_time - start_time);
	record_best_performance(rt, NUM_THREADS);
}


int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
	} else {
		//filename = "/home/zpeng/benchmarks/data/pokec/coo_tiled_bak/soc-pokec";
		filename = "/home/zpeng/benchmarks/data/skitter/coo_tiled_bak/out.skitter";
		TILE_WIDTH = 1024;
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_ends;
	unsigned *tile_offsets;
	int *is_empty_tile;
	//unsigned *nneibor;
#ifdef ONESERIAL
	//input_serial("/home/zpeng/benchmarks/data/fake/data.txt", graph_heads, graph_ends);
	input_serial("/home/zpeng/benchmarks/data/fake/mun_twitter", graph_heads, graph_ends);
#else
	input(
		filename, 
		graph_heads, 
		graph_ends,
		tile_offsets,
		is_empty_tile);
#endif

	// Connected Component
	int *graph_active = (int *) malloc(NNODES * sizeof(int));
	int *graph_updating_active = (int *) malloc(NNODES * sizeof(int));
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
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
	ROW_STEP = 16;
	for (int cz = 0; cz < 3; ++cz) {
	for (unsigned i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		for (int r = 0; r < 3; ++r) {
		for (unsigned k = 0; k < NNODES; ++k) {
			graph_active[k] = 1;
		}
		memset(graph_updating_active, 0, NNODES * sizeof(int));
		for (unsigned k = 0; k < SIDE_LENGTH; ++k) {
			is_active_side[k] = 1;
		}
		memset(is_updating_active_side, 0, SIDE_LENGTH * sizeof(int));
		for (unsigned k = 0; k < NNODES; ++k) {
			graph_component[k] = k;
		}
		//sleep(10);
		cc(
			graph_heads, 
			graph_ends, 
			tile_offsets,
			graph_active, 
			graph_updating_active, 
			is_active_side,
			is_updating_active_side,
			is_empty_tile,
			graph_component);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
		}
	}
	}
	fclose(time_out);
	print_best_performance();
#ifdef ONEDEBUG
	print(graph_component);
#endif

	// Free memory
	free(graph_heads);
	free(graph_ends);
	free(tile_offsets);
	free(graph_active);
	free(graph_updating_active);
	free(is_active_side);
	free(is_updating_active_side);
	free(is_empty_tile);
	free(graph_component);

	return 0;
}
