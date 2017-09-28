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
#include <immintrin.h>
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH; // Width of tile
unsigned SIDE_LENGTH; // Number of rows of tiles
unsigned NUM_TILES; // Number of tiles
unsigned ROW_STEP; // Number of rows of tiles in a Group
unsigned SIZE_BUFFER_MAX; // Buffer size for every thread (stripe)

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
		tile_offsets[i] = 2 * offset;
	}
	fclose(fin);
	graph_heads = (unsigned *) malloc(2 * NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(2 * NEDGES * sizeof(unsigned));
	is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
	memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			if (tile_offsets[i] == tile_offsets[i + 1]) {
				is_empty_tile[i] = 1;
			}
		} else {
			if (tile_offsets[i] == 2 * NEDGES) {
				is_empty_tile[i] = 1;
			}
		}
	}
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
	//for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
	//	unsigned head = graph_heads[edge_i];
	//	unsigned end = graph_ends[edge_i];
	//	if (0 == graph_active[end]) {
	//		continue;
	//	}
	//	if (graph_component[head] > graph_component[end]) {
	//		graph_updating_active[head] = 1;
	//		graph_component[head] = graph_component[end];
	//	}
	//}
	unsigned edge_i;
	for (edge_i = edge_i_start; edge_i + NUM_P_INT <= edge_i_bound; edge_i += NUM_P_INT) {
		__m512i head_v = _mm512_load_epi32(graph_heads + edge_i);
		__m512i end_v = _mm512_load_epi32(graph_ends + edge_i);
		__m512i active_v = _mm512_i32gather_epi32(end_v, graph_active, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			continue;
		}
		__m512i component_head_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), is_active_m, head_v, graph_component, sizeof(unsigned));
		__m512i component_end_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), is_active_m, end_v, graph_component, sizeof(unsigned));
		__mmask16 head_gt_end_m = _mm512_cmpgt_epi32_mask(component_head_v, component_end_v);
		_mm512_mask_i32scatter_epi32(graph_updating_active, head_gt_end_m, head_v, _mm512_set1_epi32(1), sizeof(int));
		_mm512_mask_i32scatter_epi32(graph_component, head_gt_end_m, head_v, component_end_v, sizeof(unsigned));
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
					unsigned *heads_buffer,
					unsigned *ends_buffer,
					unsigned *graph_component,
					const unsigned &start_row_index,
					const unsigned &bound_row_index)
{
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned col_id = 0; col_id < SIDE_LENGTH; ++col_id) {
		unsigned tid = omp_get_thread_num();
		unsigned *heads_buffer_base = heads_buffer + tid * SIZE_BUFFER_MAX;
		unsigned *ends_buffer_base = ends_buffer + tid * SIZE_BUFFER_MAX;
		unsigned size_buffer = 0;
		unsigned capacity = SIZE_BUFFER_MAX;
		for (unsigned row_id = start_row_index; row_id < bound_row_index; ++row_id) {
			unsigned tile_id = row_id * SIDE_LENGTH + col_id;
			if (is_empty_tile[tile_id]) {
				continue;
			}
			// Load to buffer
			unsigned edge_i = tile_offsets[tile_id];
			unsigned remain;
			if (NUM_TILES - 1 != tile_id) {
				remain = tile_offsets[tile_id + 1] - edge_i;
			} else {
				remain = 2 * NEDGES - edge_i;
			}
			while (remain != 0) {
				if (capacity > 0) {
					if (capacity > remain) {
						// Put all remain into the buffer
						memcpy(heads_buffer_base + size_buffer, graph_heads + edge_i, remain * sizeof(unsigned));
						memcpy(ends_buffer_base + size_buffer, graph_ends + edge_i, remain * sizeof(unsigned));
						edge_i += remain;
						capacity -= remain;
						size_buffer += remain;
						remain = 0;
					} else {
						// Fill the buffer to full
						memcpy(heads_buffer_base + size_buffer, graph_heads + edge_i, capacity * sizeof(unsigned));
						memcpy(ends_buffer_base + size_buffer, graph_ends + edge_i, capacity * sizeof(unsigned));
						edge_i += capacity;
						remain -= capacity;
						size_buffer += capacity;
						capacity = 0;
					}
				} else {
					// Process the full buffer
					cc_kernel(
							heads_buffer_base,
							ends_buffer_base,
							graph_active, 
							graph_updating_active,
							is_active_side,
							is_updating_active_side,
							graph_component,
							0, 
							size_buffer);
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}

			//unsigned bound_edge_i;
			//if (NUM_TILES - 1 != tile_id) {
			//	bound_edge_i = tile_offsets[tile_id + 1];
			//} else {
			//	bound_edge_i = NEDGES;
			//}
			//cc_kernel(
			//	graph_heads, 
			//	graph_ends, 
			//	graph_active, 
			//	graph_updating_active,
			//	is_active_side,
			//	is_updating_active_side,
			//	graph_component,
			//	tile_offsets[tile_id], 
			//	bound_edge_i);
		}
		// Process the remains in buffer
		cc_kernel(
				heads_buffer_base,
				ends_buffer_base,
				graph_active, 
				graph_updating_active,
				is_active_side,
				is_updating_active_side,
				graph_component,
				0, 
				size_buffer);
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
	unsigned *heads_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned *ends_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
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
				heads_buffer,
				ends_buffer,
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
			heads_buffer,
			ends_buffer,
			graph_component,
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
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
	_mm_free(heads_buffer);
	_mm_free(ends_buffer);
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
	SIZE_BUFFER_MAX = 512;
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
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
	free(tile_offsets);
	free(graph_active);
	free(graph_updating_active);
	free(is_active_side);
	free(is_updating_active_side);
	free(is_empty_tile);
	free(graph_component);

	return 0;
}
