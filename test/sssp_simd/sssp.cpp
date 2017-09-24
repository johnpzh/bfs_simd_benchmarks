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
#include <immintrin.h>
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

int nnodes, nedges;
unsigned NUM_THREADS; // Number of threads
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
	//for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
	//	unsigned head = graph_heads[edge_i];
	//	if (0 == graph_active[head]) {
	//		continue;
	//	}
	//	unsigned end = graph_ends[edge_i];
	//	if (-1 == dists[end] || dists[head] + 1 < dists[end]) {
	//		dists[end] = dists[head] + 1;
	//		graph_updating_active[end] = 1;
	//		is_updating_active_side[end/TILE_WIDTH] = 1;
	//	}
	//}
	unsigned edge_i;
	for (edge_i = edge_i_start; edge_i + NUM_P_INT <= edge_i_bound; edge_i += NUM_P_INT) {
		__m512i head_v = _mm512_load_epi32(graph_heads + edge_i);
		__m512i active_flag_v = _mm512_i32gather_epi32(head_v, graph_active, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			continue;
		}
		__m512i end_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, graph_ends + edge_i);
		__m512i dists_end_v = _mm512_i32gather_epi32(end_v, dists, sizeof(unsigned));
		__m512i dists_head_v = _mm512_i32gather_epi32(head_v, dists, sizeof(unsigned));
		__m512i dists_tmp_v = _mm512_add_epi32(dists_head_v, _mm512_set1_epi32(1));
		__mmask16 is_minusone_m = _mm512_cmpeq_epi32_mask(dists_end_v, _mm512_set1_epi32(-1));
		__mmask16 is_shorter_m = _mm512_cmplt_epi32_mask(dists_tmp_v, dists_end_v);
		__mmask16 need_update_m = is_minusone_m | is_shorter_m;
		if (!need_update_m) {
			continue;
		}
		_mm512_mask_i32scatter_epi32(dists, need_update_m, end_v, dists_tmp_v, sizeof(unsigned));
		_mm512_mask_i32scatter_epi32(graph_updating_active, need_update_m, end_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(end_v, TILE_WIDTH_v);
		_mm512_mask_i32scatter_epi32(is_updating_active_side, need_update_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}

	__m512i edge_i_v = _mm512_set_epi32(edge_i + 15, edge_i + 14, edge_i + 13, edge_i + 12,\
			edge_i + 11, edge_i + 10, edge_i + 9, edge_i + 8,\
			edge_i + 7, edge_i + 6, edge_i + 5, edge_i + 4,\
			edge_i + 3, edge_i + 2, edge_i + 1, edge_i);
	__m512i size_buffer_v = _mm512_set1_epi32(edge_i_bound);
	__mmask16 in_range_m = _mm512_cmplt_epi32_mask(edge_i_v, size_buffer_v);
	__m512i head_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, graph_heads + edge_i);
	__m512i active_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, head_v, graph_active, sizeof(int));
	__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
	if (!is_active_m) {
		return;
	}
	__m512i end_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, graph_ends + edge_i);
	__m512i dists_end_v = _mm512_i32gather_epi32(end_v, dists, sizeof(unsigned));
	__m512i dists_head_v = _mm512_i32gather_epi32(head_v, dists, sizeof(unsigned));
	__m512i dists_tmp_v = _mm512_add_epi32(dists_head_v, _mm512_set1_epi32(1));
	__mmask16 is_minusone_m = _mm512_cmpeq_epi32_mask(dists_end_v, _mm512_set1_epi32(-1));
	__mmask16 is_shorter_m = _mm512_cmplt_epi32_mask(dists_tmp_v, dists_end_v);
	__mmask16 need_update_m = is_minusone_m | is_shorter_m;
	if (!need_update_m) {
		return;
	}
	_mm512_mask_i32scatter_epi32(dists, need_update_m, end_v, dists_tmp_v, sizeof(unsigned));
	_mm512_mask_i32scatter_epi32(graph_updating_active, need_update_m, end_v, _mm512_set1_epi32(1), sizeof(int));
	__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
	__m512i side_id_v = _mm512_div_epi32(end_v, TILE_WIDTH_v);
	_mm512_mask_i32scatter_epi32(is_updating_active_side, need_update_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
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
					int *dists, 
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
				remain = nedges - edge_i;
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
					sssp_kernel(
							heads_buffer_base,
							ends_buffer_base,
							graph_active, 
							graph_updating_active,
							is_active_side,
							is_updating_active_side,
							dists, 
							0, 
							size_buffer);
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}
		}
		// Process the remains in buffer
		sssp_kernel(
				heads_buffer_base,
				ends_buffer_base,
				graph_active, 
				graph_updating_active,
				is_active_side,
				is_updating_active_side,
				dists, 
				0, 
				size_buffer);
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
	unsigned *heads_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned *ends_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	double start_time = omp_get_wtime();
	int stop = 0;
	int vcount = 1;//test
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
			heads_buffer,
			ends_buffer,
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
			//for (unsigned vertex_id = side_id * TILE_WIDTH; vertex_id < bound_vertex_id; ++vertex_id) {
			//	if (1 == graph_updating_active[vertex_id]) {
			//		graph_updating_active[vertex_id] = 0;
			//		graph_active[vertex_id] = 1;
			//	} else {
			//		graph_active[vertex_id] = 0;
			//	}
			//}
			unsigned vertex_id;
			for (vertex_id = side_id * TILE_WIDTH;
					vertex_id + NUM_P_INT <= bound_vertex_id;
					vertex_id += NUM_P_INT) {
				__m512i updating_flag_v = _mm512_loadu_si512(graph_updating_active + vertex_id);
				__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
				if (!is_updating_m) {
					continue;
				}
				_mm512_mask_storeu_epi32(graph_updating_active + vertex_id, is_updating_m, _mm512_set1_epi32(0));
				_mm512_storeu_si512(graph_active + vertex_id, updating_flag_v);
				vcount += 16;//test
			}
			__m512i vertex_id_v = _mm512_set_epi32(
											vertex_id + 15, vertex_id + 14, vertex_id + 13, vertex_id + 12,\
											vertex_id + 11, vertex_id + 10, vertex_id + 9, vertex_id + 8,\
											vertex_id + 7, vertex_id + 6, vertex_id + 5, vertex_id + 4,\
											vertex_id + 3, vertex_id + 2, vertex_id + 1, vertex_id);
			__m512i bound_vertex_id_v = _mm512_set1_epi32(bound_vertex_id);
			__mmask16 in_range_m = _mm512_cmplt_epi32_mask(vertex_id_v, bound_vertex_id_v);
			__m512i updating_flag_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), in_range_m, graph_updating_active + vertex_id);
			__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
			if (!is_updating_m) {
				continue;
			}
			_mm512_mask_storeu_epi32(graph_updating_active + vertex_id, is_updating_m, _mm512_set1_epi32(0));
			_mm512_storeu_si512(graph_active + vertex_id, updating_flag_v);
			vcount += 16;//test
		}
		printf("vcount: %d\n", vcount);//test
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
	int *graph_updating_active = (int *) _mm_malloc(nnodes * sizeof(int), ALIGNED_BYTES);
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
	SIZE_BUFFER_MAX = 512;
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
	_mm_free(graph_updating_active);
	free(is_active_side);
	free(is_updating_active_side);

	return 0;
}
