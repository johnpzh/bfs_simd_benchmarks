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
#include <immintrin.h>
#include <hbwmalloc.h>
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;
using std::vector;

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

unsigned KCORE;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_ends, 
		unsigned *&graph_degrees,
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
	tile_offsets = (unsigned *) hbw_malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		//fscanf(fin, "%u", tile_offsets + i);
		unsigned offset;
		fscanf(fin, "%u", &offset);
		tile_offsets[i] = offset;
	}
	fclose(fin);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	is_empty_tile = (int *) hbw_malloc(sizeof(int) * NUM_TILES);
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
	graph_degrees = (unsigned *) hbw_malloc(NNODES * sizeof(unsigned));
	memset(graph_degrees, 0, NNODES * sizeof(unsigned));
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
#pragma omp atomic
		graph_degrees[n1]++;
	}

	fclose(fin);
}
}

void input_serial(char filename[], unsigned *&graph_heads, unsigned *&graph_ends, unsigned *&graph_degrees)
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
	graph_degrees = (unsigned *) calloc(NNODES, sizeof(unsigned));
	for (unsigned i = 0; i < NEDGES; ++i) {
		unsigned head;
		unsigned end;
		fscanf(fin, "%u %u", &head, &end);
		--head;
		--end;
		graph_heads[i] = head;
		graph_ends[i] = end;
		graph_degrees[head]++;
		//graph_degrees[end]++;
	}
	fclose(fin);
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
	fprintf(foutput, "kc: %u, KCORE: %u\n", kc, KCORE);
}
//unsigned test_count = 0;//test
inline void kcore_kernel(
				unsigned *graph_heads, 
				unsigned *graph_ends,
				unsigned *graph_degrees,
				int *graph_updating_active, 
				unsigned *graph_cores,
				const unsigned &edge_i_start, 
				const unsigned &edge_i_bound)
{
	//for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
	//	unsigned head = graph_heads[edge_i];
	//	unsigned end = graph_ends[edge_i];
	//	if (graph_updating_active[head] && graph_degrees[end]) {
	//		graph_degrees[end]--;
	//		if (!graph_degrees[end]) {
	//			graph_cores[end] = KCORE - 1;
	//			//test_count++;//test
	//		}
	//	}
	//}
	unsigned edge_i;
	for (edge_i = edge_i_start; edge_i + NUM_P_INT <= edge_i_bound; edge_i += NUM_P_INT) {
		__m512i head_v = _mm512_load_epi32(graph_heads + edge_i);
		__m512i end_v = _mm512_load_epi32(graph_ends + edge_i);
		__m512i active_v = _mm512_i32gather_epi32(head_v, graph_updating_active, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_v, _mm512_set1_epi32(-1));
		__m512i end_degrees_v = _mm512_i32gather_epi32(end_v, graph_degrees, sizeof(unsigned));
		__mmask16 not_removed_m = _mm512_test_epi32_mask(end_degrees_v, _mm512_set1_epi32(-1));
		__mmask16 need_reduce_m = is_active_m & not_removed_m;
		if (need_reduce_m) {
			//__m512i subt_one_v = _mm512_set1_epi32(1);
			__m512i subt_one_v = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), need_reduce_m, 1);
			__m512i conflict_end = _mm512_conflict_epi32(end_v);
			//__m512i conflict_end = _mm512_mask_conflict_epi32(_mm512_set1_epi32(0), need_reduce_m, end_v);
			__mmask16 todo_mask = _mm512_test_epi32_mask(conflict_end, _mm512_set1_epi32(-1));
			//__mmask16 todo_mask = _mm512_mask_test_epi32_mask(need_reduce_m, conflict_end, _mm512_set1_epi32(-1));
			if (todo_mask) {
				__m512i lz = _mm512_lzcnt_epi32(conflict_end);
				__m512i lid = _mm512_sub_epi32(_mm512_set1_epi32(31), lz);
				while (todo_mask) {
					__m512i todo_bcast = _mm512_broadcastmw_epi32(todo_mask);
					__mmask16 now_mask = _mm512_mask_testn_epi32_mask(todo_mask, conflict_end, todo_bcast);
					__m512i subt_one_v_perm = _mm512_mask_permutexvar_epi32(_mm512_undefined_epi32(), now_mask, lid, subt_one_v);
					subt_one_v = _mm512_mask_add_epi32(subt_one_v, now_mask, subt_one_v, subt_one_v_perm);
					todo_mask = _mm512_kxor(todo_mask, now_mask);
				}
			}

			__m512i end_degrees_results_v = _mm512_mask_sub_epi32(end_degrees_v, need_reduce_m, end_degrees_v, subt_one_v);
			_mm512_mask_i32scatter_epi32(graph_degrees, need_reduce_m, end_v, end_degrees_results_v, sizeof(unsigned));
		}
	}
	__m512i edge_i_v = _mm512_set_epi32(edge_i + 15, edge_i + 14, edge_i + 13, edge_i + 12,\
										edge_i + 11, edge_i + 10, edge_i + 9, edge_i + 8,\
										edge_i + 7, edge_i + 6, edge_i + 5, edge_i + 4,\
										edge_i + 3, edge_i + 2, edge_i + 1, edge_i);
	__m512i edge_i_bound_v = _mm512_set1_epi32(edge_i_bound);
	__mmask16 in_range_m = _mm512_cmplt_epi32_mask(edge_i_v, edge_i_bound_v);
	__m512i head_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, graph_heads + edge_i);
	__m512i end_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, graph_ends + edge_i);
	__m512i active_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, head_v, graph_updating_active, sizeof(int));
	__mmask16 is_active_m = _mm512_test_epi32_mask(active_v, _mm512_set1_epi32(-1));
	__m512i end_degrees_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, end_v, graph_degrees, sizeof(unsigned));
	__mmask16 not_removed_m = _mm512_test_epi32_mask(end_degrees_v, _mm512_set1_epi32(-1));
	__mmask16 need_reduce_m = is_active_m & not_removed_m;
	if (need_reduce_m) {
		//__m512i subt_one_v = _mm512_set1_epi32(1);
		__m512i subt_one_v = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), need_reduce_m, 1);
		__m512i conflict_end = _mm512_conflict_epi32(end_v);
		//__m512i conflict_end = _mm512_mask_conflict_epi32(_mm512_set1_epi32(0), need_reduce_m, end_v);
		__mmask16 todo_mask = _mm512_test_epi32_mask(conflict_end, _mm512_set1_epi32(-1));
		//__mmask16 todo_mask = _mm512_mask_test_epi32_mask(need_reduce_m, conflict_end, _mm512_set1_epi32(-1));
		if (todo_mask) {
			__m512i lz = _mm512_lzcnt_epi32(conflict_end);
			__m512i lid = _mm512_sub_epi32(_mm512_set1_epi32(31), lz);
			while (todo_mask) {
				__m512i todo_bcast = _mm512_broadcastmw_epi32(todo_mask);
				__mmask16 now_mask = _mm512_mask_testn_epi32_mask(todo_mask, conflict_end, todo_bcast);
				__m512i subt_one_v_perm = _mm512_mask_permutexvar_epi32(_mm512_undefined_epi32(), now_mask, lid, subt_one_v);
				subt_one_v = _mm512_mask_add_epi32(subt_one_v, now_mask, subt_one_v, subt_one_v_perm);
				todo_mask = _mm512_kxor(todo_mask, now_mask);
			}
		}

		__m512i end_degrees_results_v = _mm512_mask_sub_epi32(end_degrees_v, need_reduce_m, end_degrees_v, subt_one_v);
		_mm512_mask_i32scatter_epi32(graph_degrees, need_reduce_m, end_v, end_degrees_results_v, sizeof(unsigned));
	}
}
inline void scheduler(
					unsigned *graph_heads, 
					unsigned *graph_ends, 
					unsigned *graph_degrees,
					unsigned *tile_offsets,
					int *graph_updating_active,
					int *is_empty_tile,
					unsigned *heads_buffer,
					unsigned *ends_buffer,
					unsigned *graph_cores,
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
				remain = NEDGES - edge_i;
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
					kcore_kernel(
							heads_buffer_base,
							ends_buffer_base,
							graph_degrees,
							graph_updating_active,
							graph_cores,
							0, 
							size_buffer);
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}
			////bfs_kernel();
			//unsigned bound_edge_i;
			//if (NUM_TILES - 1 != tile_id) {
			//	bound_edge_i = tile_offsets[tile_id + 1];
			//} else {
			//	bound_edge_i = NEDGES;
			//}
			//kcore_kernel(
			//	graph_heads, 
			//	graph_ends, 
			//	graph_degrees,
			//	graph_updating_active,
			//	graph_cores,
			//	tile_offsets[tile_id], 
			//	bound_edge_i);
		}
		// Process the remains in buffer
		kcore_kernel(
				heads_buffer_base,
				ends_buffer_base,
				graph_degrees,
				graph_updating_active,
				graph_cores,
				0, 
				size_buffer);
	}
}
void kcore(
		unsigned *graph_heads, 
		unsigned *graph_ends, 
		unsigned *graph_degrees,
		unsigned *tile_offsets,
		int *graph_updating_active,
		int *is_updating_active_side,
		int *is_empty_tile,
		unsigned *graph_cores)
{
	omp_set_num_threads(NUM_THREADS);
	unsigned *heads_buffer;
   	hbw_posix_memalign((void **) &heads_buffer, ALIGNED_BYTES, sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS);
	unsigned *ends_buffer;
   	hbw_posix_memalign((void **) &ends_buffer, ALIGNED_BYTES, sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS);
	double start_time = omp_get_wtime();
	int stop = 0;
	//test_count = 0;
	while (!stop) {
		stop = 1;
		int has_remove = 1;
		KCORE++;
		while (has_remove) {
			double ts = omp_get_wtime();
			has_remove = 0;
//#pragma omp parallel for schedule(dynamic, 1)
#pragma omp parallel for
			for (unsigned i = 0; i < NNODES; ++i) {
				if (graph_degrees[i]) {
					stop = 0;
					if(graph_degrees[i] < KCORE) {
						graph_updating_active[i] = 1;
						is_updating_active_side[i/TILE_WIDTH] = 1;
						graph_degrees[i] = 0;
						graph_cores[i] = KCORE - 1;
						//test_count++;//test
						has_remove = 1;
					}
				}
			}
			double ts2 = omp_get_wtime();
			//printf("time for vertices: %lf\n", ts2 - ts);
			unsigned side_id;
			for (side_id = 0; side_id + ROW_STEP <= SIDE_LENGTH; ) {
				if (!is_updating_active_side[side_id]) {
					++side_id;
					continue;
				}
				scheduler(
						graph_heads, 
						graph_ends, 
						graph_degrees,
						tile_offsets,
						graph_updating_active,
						is_empty_tile,
						heads_buffer,
						ends_buffer,
						graph_cores,
						side_id,
						side_id + ROW_STEP);
				side_id += ROW_STEP;
			}
			scheduler(
					graph_heads, 
					graph_ends, 
					graph_degrees,
					tile_offsets,
					graph_updating_active,
					is_empty_tile,
					heads_buffer,
					ends_buffer,
					graph_cores,
					side_id,
					SIDE_LENGTH);
			//kcore_kernel(
			//		graph_heads, 
			//		graph_ends, 
			//		graph_degrees,
			//		graph_updating_active, 
			//		0, 
			//		NEDGES,
			//		graph_cores);
			//printf("time for edges: %lf\n", omp_get_wtime() - ts2);
			memset(graph_updating_active, 0, NNODES * sizeof(int));
			memset(is_updating_active_side, 0, SIDE_LENGTH * sizeof(int));
		}
		printf("KCORE: %u\n", KCORE);//test
	}
	KCORE -= 2;

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
	hbw_free(heads_buffer);
	hbw_free(ends_buffer);
}


int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
	} else {
		//filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		filename = "/home/zpeng/benchmarks/data/skitter/out.skitter";
		TILE_WIDTH = 1024;
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_ends;
	unsigned *graph_degrees;
	unsigned *tile_offsets;
	int *is_empty_tile;
#ifdef ONESERIAL
	//input_serial("/home/zpeng/benchmarks/data/fake/data.txt", graph_heads, graph_ends, graph_degrees);
	//input_serial("/home/zpeng/benchmarks/data/fake/mun_twitter", graph_heads, graph_ends,graph_degrees);
	input_serial("/home/zpeng/benchmarks/data/zebra/out.zebra", graph_heads, graph_ends,graph_degrees);
#else
	input(
		filename, 
		graph_heads, 
		graph_ends, 
		graph_degrees,
		tile_offsets,
		is_empty_tile);
#endif

	// K-core
	int *graph_updating_active = (int *) hbw_malloc(NNODES * sizeof(int));
	int *is_updating_active_side = (int *) hbw_malloc(sizeof(int) * SIDE_LENGTH);
	unsigned *graph_cores = (unsigned *) hbw_malloc(NNODES * sizeof(unsigned));
	unsigned *graph_degrees_bak = (unsigned *) hbw_malloc(NNODES * sizeof(unsigned));
	memcpy(graph_degrees_bak, graph_degrees, NNODES * sizeof(unsigned));
	
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned run_count = 1;
	printf("Start K-core...\n");
#else
	unsigned run_count = 9;
	printf("Start K-core...\n");//test
#endif
	//for (unsigned s = 1; s < 2048; s *= 2) {
	//ROW_STEP = s;
	//printf("ROW_STEP: %u\n", ROW_STEP);
	ROW_STEP = 16;
	SIZE_BUFFER_MAX = 512;
	for (unsigned i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		memset(graph_updating_active, 0, NNODES * sizeof(int));
		memset(is_updating_active_side, 0, SIDE_LENGTH * sizeof(int));
		for (unsigned k = 0; k < NNODES; ++k) {
			graph_cores[k] = 0;
		}
		KCORE = 0;
		memcpy(graph_degrees, graph_degrees_bak, NNODES * sizeof(unsigned));
		//sleep(10);
		kcore(
			graph_heads, 
			graph_ends, 
			graph_degrees,
			tile_offsets,
			graph_updating_active,
			is_updating_active_side,
			is_empty_tile,
			graph_cores);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	//}
	fclose(time_out);
#ifdef ONEDEBUG
	print(graph_cores);
#endif

	// Free memory
	free(graph_heads);
	free(graph_ends);
	hbw_free(graph_degrees);
	hbw_free(tile_offsets);
	hbw_free(graph_degrees_bak);
	hbw_free(graph_updating_active);
	hbw_free(is_updating_active_side);
	hbw_free(is_empty_tile);
	hbw_free(graph_cores);

	return 0;
}
