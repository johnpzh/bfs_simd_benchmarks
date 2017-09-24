#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <unistd.h>
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;


int nnodes, nedges;
unsigned NUM_THREADS; // Number of threads
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
	//string prefix = string(filename) + "_untiled";
	string prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%u %u", &nnodes, &nedges);
	fclose(fin);
	if (nnodes % TILE_WIDTH) {
		SIDE_LENGTH = nnodes / TILE_WIDTH + 1;
	} else {
		SIDE_LENGTH = nnodes / TILE_WIDTH;
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
		fscanf(fin, "%u", tile_offsets + i);
	}
	fclose(fin);
	graph_heads = (unsigned *) malloc(nedges * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(nedges * sizeof(unsigned));
	is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
	memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			if (tile_offsets[i] == tile_offsets[i + 1]) {
				is_empty_tile[i] = 1;
			}
		} else {
			if (tile_offsets[i] == nedges) {
				is_empty_tile[i] = 1;
			}
		}
	}
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
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = nedges;
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

void input_serial(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_ends)
{
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

inline void sssp_kernel(
				unsigned *graph_heads, 
				unsigned *graph_ends, 
				int *graph_active, 
				int *graph_updating_active,
				int *is_active_side,
				int *is_updating_active_side,
				int *dists, 
				const unsigned &edge_i_start, 
				const unsigned &edge_i_bound)
{
	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		if (0 == graph_active[head]) {
			continue;
		}
		unsigned end = graph_ends[edge_i];
		if (-1 == dists[end] || dists[head] + 1 < dists[end]) {
			dists[end] = dists[head] + 1;
			graph_updating_active[end] = 1;
			is_updating_active_side[end/TILE_WIDTH] = 1;
		}
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
					int *dists, 
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
				bound_edge_i = nedges;
			}
			sssp_kernel(
				graph_heads, 
				graph_ends, 
				graph_active, 
				graph_updating_active,
				is_active_side,
				is_updating_active_side,
				dists, 
				tile_offsets[tile_id], 
				bound_edge_i);
		}
	}
}

void sssp(
		unsigned *graph_heads, 
		unsigned *graph_ends, 
		unsigned *tile_offsets,
		int *graph_active, 
		int *graph_updating_active,
		int *is_active_side,
		int *is_updating_active_side,
		int *is_empty_tile,
		int *dists)
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
				dists, 
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
			dists, 
			side_id,
			SIDE_LENGTH);
#pragma omp parallel for
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
				bound_vertex_id = nnodes;
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
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
}


int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/coo_tiled_bak/soc-pokec";
		TILE_WIDTH = 1024;
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_ends;
	unsigned *tile_offsets;
	int *is_empty_tile;
	//unsigned *nneibor;
#ifdef ONESERIAL
	input_serial("data.txt", graph_heads, graph_ends);
#else
	input(
		filename, 
		graph_heads, 
		graph_ends, 
		tile_offsets,
		is_empty_tile);
#endif

	// SSSP
	int *distances = (int *) malloc(nnodes * sizeof(int));
	int *graph_active = (int *) malloc(nnodes * sizeof(int));
	int *graph_updating_active = (int *) malloc(nnodes * sizeof(int));
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned source = 0;
	
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("SSSP starts...\n");
	unsigned run_count = 2;
#else
	unsigned run_count = 9;
#endif
	ROW_STEP = 16;
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		memset(distances, -1, nnodes * sizeof(int));
		distances[source] = 0;
		memset(graph_active, 0, nnodes * sizeof(int));
		graph_active[source] = 1;
		memset(graph_updating_active, 0, nnodes * sizeof(int));
		memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
		is_active_side[source/TILE_WIDTH] = 1;
		memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);

		//sleep(10);
		sssp(
			graph_heads, 
			graph_ends, 
			tile_offsets,
			graph_active, 
			graph_updating_active,
			is_active_side,
			is_updating_active_side,
			is_empty_tile,
			distances);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);
#ifdef ONEDEBUG
	print(distances);
#endif

	// Free memory
	free(graph_heads);
	free(graph_ends);
	free(tile_offsets);
	free(is_empty_tile);
	free(distances);
	free(graph_active);
	free(graph_updating_active);
	free(is_active_side);
	free(is_updating_active_side);

	return 0;
}
