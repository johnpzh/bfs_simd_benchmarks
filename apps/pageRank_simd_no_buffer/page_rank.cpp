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
#include <immintrin.h>
#include <unistd.h>
#include "../../include/peg_util.h"
//#include "peg.h"
//#include <papi.h>

using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
using std::to_string;

#define DUMP 0.85
#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

unsigned NNODES, NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH;
unsigned CHUNK_SIZE;
unsigned SIZE_BUFFER_MAX;
unsigned ROW_STEP;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


/////////////////////
//void page_rank(unsigned *n1s, unsigned *n2s, unsigned *graph_degrees, float *rank, float *sum) {
//
//	//for(int i=0;i<10;i++) {
//	double start_time = omp_get_wtime();
//
//#pragma omp parallel for num_threads(NUM_THREADS)
//	for(unsigned j=0;j<NEDGES;j++) {
//		int n1 = n1s[j];
//		int n2 = n2s[j];
//#pragma omp atomic
//		sum[n2] += rank[n1]/graph_degrees[n1];
//	}
//	//cout << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0 << endl;
//	double end_time = omp_get_wtime();
//	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
//
//	for(unsigned j = 0; j < NNODES; j++) {
//		rank[j] = (1 - DUMP) / NNODES + DUMP * sum[j]; 	
//	}
//	//}
//}
////////////////////////////

//const __m512i one_v = _mm512_set1_epi32(1);
//const __m512i zero_v = _mm512_set1_epi32(0);
//const __m512i minusone_v = _mm512_set1_epi32(-1);

// Scan the data, accumulate the values with the same index.
// Then, store the cumulative sum to the last element in the data with the same index.
inline void scan_for_gather_add_scatter_conflict_safe_epi32(
												__m512i &data,
												__m512i indices)
{
	__m512i cd = _mm512_conflict_epi32(indices);
	__mmask16 todo_mask = _mm512_test_epi32_mask(cd, _mm512_set1_epi32(-1));
	if (todo_mask) {
		__m512i lz = _mm512_lzcnt_epi32(cd);
		__m512i lid = _mm512_sub_epi32(_mm512_set1_epi32(31), lz);
		while (todo_mask) {
			__m512i todo_bcast = _mm512_broadcastmw_epi32(todo_mask);
			__mmask16 now_mask = _mm512_mask_testn_epi32_mask(todo_mask, cd, todo_bcast);
			__m512i data_perm = _mm512_mask_permutexvar_epi32(_mm512_undefined_epi32(), now_mask, lid, data);
			data = _mm512_mask_add_epi32(data, now_mask, data, data_perm);
			todo_mask = _mm512_kxor(todo_mask, now_mask);
		}
	} 
}

// Scan the data, accumulate the values with the same index.
// Then, store the cumulative sum to the last element in the data with the same index.
inline void scan_for_gather_add_scatter_conflict_safe_ps(
												__m512 &data,
												__m512i indices)
{
	__m512i cd = _mm512_conflict_epi32(indices);
	__mmask16 todo_mask = _mm512_test_epi32_mask(cd, _mm512_set1_epi32(-1));
	if (todo_mask) {
		__m512i lz = _mm512_lzcnt_epi32(cd);
		__m512i lid = _mm512_sub_epi32(_mm512_set1_epi32(31), lz);
		while (todo_mask) {
			__m512i todo_bcast = _mm512_broadcastmw_epi32(todo_mask);
			__mmask16 now_mask = _mm512_mask_testn_epi32_mask(todo_mask, cd, todo_bcast);
			__m512 data_perm = _mm512_mask_permutexvar_ps(_mm512_undefined_ps(), now_mask, lid, data);
			data = _mm512_mask_add_ps(data, now_mask, data, data_perm);
			todo_mask = _mm512_kxor(todo_mask, now_mask);
		}
	} 
}

inline void kernel_pageRank(
		unsigned *n1_buffer,
		unsigned *n2_buffer,
		const unsigned &size_buffer,
		float *sum,
		float *rank,
		unsigned *graph_degrees)
{
	unsigned remainder = size_buffer % NUM_P_INT;
	unsigned bound_edge_i = size_buffer - remainder;
	for (unsigned edge_i = 0; edge_i < bound_edge_i; edge_i += NUM_P_INT) {
		//bot_simd_util.record(NUM_P_INT, NUM_P_INT);
		__m512i n1_v = _mm512_load_epi32(n1_buffer + edge_i);
		__m512i n2_v = _mm512_load_epi32(n2_buffer + edge_i);

		__m512 rank_v = _mm512_i32gather_ps(n1_v, rank, sizeof(float));
		__m512i graph_degrees_vi = _mm512_i32gather_epi32(n1_v, graph_degrees, sizeof(int));
		__m512 graph_degrees_v = _mm512_cvtepi32_ps(graph_degrees_vi);
		__m512 tmp_sum = _mm512_div_ps(rank_v, graph_degrees_v);
		scan_for_gather_add_scatter_conflict_safe_ps(tmp_sum, n2_v);
		__m512 sum_n2_v = _mm512_i32gather_ps(n2_v, sum, sizeof(float));
		tmp_sum = _mm512_add_ps(tmp_sum, sum_n2_v);
		_mm512_i32scatter_ps(sum, n2_v, tmp_sum, sizeof(float));
	}

	if (remainder > 0) {
		//bot_simd_util.record(0, remainder);
		//__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		//__m512i n1_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, n1_buffer + bound_edge_i);
		//__m512i n2_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, n2_buffer + bound_edge_i);

		//__m512 rank_v = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), in_range_m, n1_v, rank, sizeof(float));
		//__m512i graph_degrees_vi = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), in_range_m, n1_v, graph_degrees, sizeof(int));
		//__m512 graph_degrees_v = _mm512_cvtepi32_ps(graph_degrees_vi);
		//__m512 tmp_sum = _mm512_mask_div_ps(_mm512_set1_ps(0), in_range_m, rank_v, graph_degrees_v);
		//scan_for_gather_add_scatter_conflict_safe_ps(tmp_sum, n2_v);
		//__m512 sum_n2_v = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), in_range_m, n2_v, sum, sizeof(float));
		//tmp_sum = _mm512_mask_add_ps(_mm512_undefined_ps(), in_range_m, tmp_sum, sum_n2_v);
		//_mm512_mask_i32scatter_ps(sum, in_range_m, n2_v, tmp_sum, sizeof(float));

		for (unsigned edge_i = bound_edge_i; edge_i < size_buffer; ++edge_i) {
			unsigned n1 = n1_buffer[edge_i];
			unsigned n2 = n2_buffer[edge_i];
			sum[n2] += rank[n1] / graph_degrees[n1];
		}
	}
}

inline void scheduler(
		unsigned row_index,
		unsigned tile_step,
		unsigned *graph_heads,
		unsigned *graph_tails,
		unsigned *n1_buffer,
		unsigned *n2_buffer,
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		float *sum,
		float *rank,
		unsigned *graph_degrees,
		const unsigned &side_length)
{
	unsigned *sizes_buffers = (unsigned *) calloc(NUM_THREADS, sizeof(unsigned));
	unsigned bound_row_id = row_index + tile_step;
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
	for (unsigned col_id = 0; col_id < side_length; ++col_id) {
		unsigned tid = omp_get_thread_num();
		unsigned *n1_buffer_base = n1_buffer + tid * SIZE_BUFFER_MAX;
		unsigned *n2_buffer_base = n2_buffer + tid * SIZE_BUFFER_MAX;
		//unsigned size_buffer = 0;
		//unsigned capacity = SIZE_BUFFER_MAX;
		unsigned size_buffer = sizes_buffers[tid];
		unsigned capacity = SIZE_BUFFER_MAX - size_buffer;
		for (unsigned row_id = row_index; row_id < bound_row_id; ++row_id) {
			unsigned tile_id = row_id * side_length + col_id;
			if (0 == tile_sizes[tile_id]) {
				continue;
			}
			// Load to buffer
			unsigned edge_i = tile_offsets[tile_id];
			unsigned remain = tile_sizes[tile_id];
			while (remain != 0) {
				if (capacity > 0) {
					if (capacity > remain) {
						// Put all remain into the buffer
						memcpy(n1_buffer_base + size_buffer, graph_heads + edge_i, remain * sizeof(unsigned));
						memcpy(n2_buffer_base + size_buffer, graph_tails + edge_i, remain * sizeof(unsigned));
						edge_i += remain;
						capacity -= remain;
						size_buffer += remain;
						remain = 0;
					} else {
						// Fill the buffer to full
						memcpy(n1_buffer_base + size_buffer, graph_heads + edge_i, capacity * sizeof(unsigned));
						memcpy(n2_buffer_base + size_buffer, graph_tails + edge_i, capacity * sizeof(unsigned));
						edge_i += capacity;
						remain -= capacity;
						size_buffer += capacity;
						capacity = 0;
					}
				} else {
					// Buffer is full already
					// Process what in buffer
					kernel_pageRank(\
							n1_buffer_base,\
							n2_buffer_base,\
							SIZE_BUFFER_MAX,\
							sum,\
							rank,\
							graph_degrees);
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}
		}
		sizes_buffers[tid] = size_buffer;
	}

	// Process the remains in buffer
#pragma omp parallel	
	{
		unsigned tid = omp_get_thread_num();
		unsigned *n1_buffer_base = n1_buffer + tid * SIZE_BUFFER_MAX;
		unsigned *n2_buffer_base = n2_buffer + tid * SIZE_BUFFER_MAX;
		unsigned size_buffer = sizes_buffers[tid];
		if (size_buffer > 0) {
			kernel_pageRank(\
					n1_buffer_base,\
					n2_buffer_base,\
					size_buffer,\
					sum,\
					rank,\
					graph_degrees);
		}
	}

	free(sizes_buffers);
}
void page_rank(\
		unsigned *graph_heads, \
		unsigned *graph_tails, \
		unsigned *graph_degrees, \
		unsigned *tile_sizes, \
		float *rank, \
		float *sum, \
		unsigned *tile_offsets, \
		unsigned num_tiles, \
		unsigned side_length)
{
	//unsigned ROW_STEP = 1;
	if (side_length < ROW_STEP) {
		printf("Error: ROW_STEP (%u) is to large, larger than side_length (%u)\n", \
				ROW_STEP, side_length);
		exit(3);
	}
	omp_set_num_threads(NUM_THREADS);
	unsigned *n1_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned *n2_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned remainder = side_length % ROW_STEP;
	unsigned bound_row_index = side_length - remainder;

	// Cache miss
	//CacheMissRate miss_rate;
	double start_time = omp_get_wtime(); // Timer Starts
	//miss_rate.measure_start();
	for (unsigned row_index = 0; row_index < bound_row_index; row_index += ROW_STEP) {
		scheduler(
				row_index,
				//row_index + ROW_STEP,
				ROW_STEP,
				graph_heads,
				graph_tails,
				n1_buffer,
				n2_buffer,
				tile_offsets,
				tile_sizes,
				sum,
				rank,
				graph_degrees,
				side_length);
	}

	if (remainder > 0) {
		scheduler(
				bound_row_index,
				remainder,
				graph_heads,
				graph_tails,
				n1_buffer,
				n2_buffer,
				tile_offsets,
				tile_sizes,
				sum,
				rank,
				graph_degrees,
				side_length);
	}


//#pragma omp parallel num_threads(64)
//	for(unsigned j = 0; j < NNODES; j++) {
//		rank[j] = (1 - DUMP) / NNODES + DUMP * sum[j]; 	
//	}
	remainder = NNODES % NUM_P_INT;
	unsigned bound_v_i = NNODES - remainder;
	__m512 dump_v = _mm512_set1_ps(DUMP);
	__m512 one_minus_dump_v = _mm512_sub_ps(_mm512_set1_ps(1.0), dump_v);
	__m512 first_term = _mm512_div_ps(one_minus_dump_v, _mm512_set1_ps((float) NNODES));
#pragma omp parallel for
	for (unsigned v_i = 0; v_i < bound_v_i; v_i += NUM_P_INT) {
		__m512 sum_v = _mm512_load_ps(sum + v_i);
		__m512 second_term = _mm512_mul_ps(dump_v, sum_v);
		_mm512_store_ps(rank + v_i, _mm512_add_ps(first_term, second_term));
	}
	if (remainder > 0) {
		unsigned short in_range_m_t = (unsigned short) 0xFFFF >> (NUM_P_INT - remainder);
		__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512 sum_v = _mm512_mask_load_ps(_mm512_undefined_ps(), in_range_m, sum + bound_v_i);
		__m512 second_term = _mm512_mask_mul_ps(_mm512_undefined_ps(), in_range_m, dump_v, sum_v);
		_mm512_mask_store_ps(rank + bound_v_i, in_range_m, 
							_mm512_mask_add_ps(_mm512_undefined_ps(), in_range_m, first_term, second_term));
	}

	double end_time = omp_get_wtime();
	//printf("%u %lf\n", NUM_THREADS, end_time - start_time);
	//bot_simd_util.print();

	_mm_free(n1_buffer);
	_mm_free(n2_buffer);

}

////////////////////////////////////////////////////////////
// No buffer
inline void scheduler_no_buffer(
		unsigned row_index,
		unsigned tile_step,
		unsigned *graph_heads,
		unsigned *graph_tails,
		//unsigned *n1_buffer,
		//unsigned *n2_buffer,
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		float *sum,
		float *rank,
		unsigned *graph_degrees,
		const unsigned &side_length)
{
	unsigned bound_row_id = row_index + tile_step;
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned col_id = 0; col_id < side_length; ++col_id) {
		for (unsigned row_id = row_index; row_id < bound_row_id; ++row_id) {
			unsigned tile_id = row_id * side_length + col_id;
			if (0 == tile_sizes[tile_id]) {
				continue;
			}
			unsigned *heads_start = graph_heads + tile_offsets[tile_id];
			unsigned *tails_start = graph_tails + tile_offsets[tile_id];
			kernel_pageRank(
					heads_start,
					tails_start,
					tile_sizes[tile_id],
					sum,
					rank,
					graph_degrees);
		}
	}

//	///////////////////////////////////////////////
//	unsigned *sizes_buffers = (unsigned *) calloc(NUM_THREADS, sizeof(unsigned));
//	unsigned bound_row_id = row_index + tile_step;
//#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
//	for (unsigned col_id = 0; col_id < side_length; ++col_id) {
//		unsigned tid = omp_get_thread_num();
//		unsigned *n1_buffer_base = n1_buffer + tid * SIZE_BUFFER_MAX;
//		unsigned *n2_buffer_base = n2_buffer + tid * SIZE_BUFFER_MAX;
//		//unsigned size_buffer = 0;
//		//unsigned capacity = SIZE_BUFFER_MAX;
//		unsigned size_buffer = sizes_buffers[tid];
//		unsigned capacity = SIZE_BUFFER_MAX - size_buffer;
//		for (unsigned row_id = row_index; row_id < bound_row_id; ++row_id) {
//			unsigned tile_id = row_id * side_length + col_id;
//			if (0 == tile_sizes[tile_id]) {
//				continue;
//			}
//			// Load to buffer
//			unsigned edge_i = tile_offsets[tile_id];
//			unsigned remain = tile_sizes[tile_id];
//			while (remain != 0) {
//				if (capacity > 0) {
//					if (capacity > remain) {
//						// Put all remain into the buffer
//						memcpy(n1_buffer_base + size_buffer, graph_heads + edge_i, remain * sizeof(unsigned));
//						memcpy(n2_buffer_base + size_buffer, graph_tails + edge_i, remain * sizeof(unsigned));
//						edge_i += remain;
//						capacity -= remain;
//						size_buffer += remain;
//						remain = 0;
//					} else {
//						// Fill the buffer to full
//						memcpy(n1_buffer_base + size_buffer, graph_heads + edge_i, capacity * sizeof(unsigned));
//						memcpy(n2_buffer_base + size_buffer, graph_tails + edge_i, capacity * sizeof(unsigned));
//						edge_i += capacity;
//						remain -= capacity;
//						size_buffer += capacity;
//						capacity = 0;
//					}
//				} else {
//					// Buffer is full already
//					// Process what in buffer
//					kernel_pageRank_no_buffer(\
//							n1_buffer_base,\
//							n2_buffer_base,\
//							SIZE_BUFFER_MAX,\
//							sum,\
//							rank,\
//							graph_degrees);
//					capacity = SIZE_BUFFER_MAX;
//					size_buffer = 0;
//				}
//			}
//		}
//		sizes_buffers[tid] = size_buffer;
//	}
//
//	// Process the remains in buffer
//#pragma omp parallel	
//	{
//		unsigned tid = omp_get_thread_num();
//		unsigned *n1_buffer_base = n1_buffer + tid * SIZE_BUFFER_MAX;
//		unsigned *n2_buffer_base = n2_buffer + tid * SIZE_BUFFER_MAX;
//		unsigned size_buffer = sizes_buffers[tid];
//		if (size_buffer > 0) {
//			kernel_pageRank_no_buffer(\
//					n1_buffer_base,\
//					n2_buffer_base,\
//					size_buffer,\
//					sum,\
//					rank,\
//					graph_degrees);
//		}
//	}
//
//	free(sizes_buffers);
}
void page_rank_no_buffer(\
		unsigned *graph_heads, \
		unsigned *graph_tails, \
		unsigned *graph_degrees, \
		unsigned *tile_sizes, \
		float *rank, \
		float *sum, \
		unsigned *tile_offsets, \
		unsigned num_tiles, \
		unsigned side_length)
{
	//unsigned ROW_STEP = 1;
	if (side_length < ROW_STEP) {
		printf("Error: ROW_STEP (%u) is to large, larger than side_length (%u)\n", \
				ROW_STEP, side_length);
		exit(3);
	}
	omp_set_num_threads(NUM_THREADS);
	//unsigned *n1_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	//unsigned *n2_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned remainder = side_length % ROW_STEP;
	unsigned bound_row_index = side_length - remainder;

	// Cache miss
	//CacheMissRate miss_rate;
	double start_time = omp_get_wtime(); // Timer Starts
	//miss_rate.measure_start();
	for (unsigned row_index = 0; row_index < bound_row_index; row_index += ROW_STEP) {
		scheduler_no_buffer(
				row_index,
				//row_index + ROW_STEP,
				ROW_STEP,
				graph_heads,
				graph_tails,
				//n1_buffer,
				//n2_buffer,
				tile_offsets,
				tile_sizes,
				sum,
				rank,
				graph_degrees,
				side_length);
	}

	if (remainder > 0) {
		scheduler_no_buffer(
				bound_row_index,
				remainder,
				graph_heads,
				graph_tails,
				//n1_buffer,
				//n2_buffer,
				tile_offsets,
				tile_sizes,
				sum,
				rank,
				graph_degrees,
				side_length);
	}


//#pragma omp parallel num_threads(64)
//	for(unsigned j = 0; j < NNODES; j++) {
//		rank[j] = (1 - DUMP) / NNODES + DUMP * sum[j]; 	
//	}
	remainder = NNODES % NUM_P_INT;
	unsigned bound_v_i = NNODES - remainder;
	__m512 dump_v = _mm512_set1_ps(DUMP);
	__m512 one_minus_dump_v = _mm512_sub_ps(_mm512_set1_ps(1.0), dump_v);
	__m512 first_term = _mm512_div_ps(one_minus_dump_v, _mm512_set1_ps((float) NNODES));
#pragma omp parallel for
	for (unsigned v_i = 0; v_i < bound_v_i; v_i += NUM_P_INT) {
		__m512 sum_v = _mm512_load_ps(sum + v_i);
		__m512 second_term = _mm512_mul_ps(dump_v, sum_v);
		_mm512_store_ps(rank + v_i, _mm512_add_ps(first_term, second_term));
	}
	if (remainder > 0) {
		unsigned short in_range_m_t = (unsigned short) 0xFFFF >> (NUM_P_INT - remainder);
		__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512 sum_v = _mm512_mask_load_ps(_mm512_undefined_ps(), in_range_m, sum + bound_v_i);
		__m512 second_term = _mm512_mask_mul_ps(_mm512_undefined_ps(), in_range_m, dump_v, sum_v);
		_mm512_mask_store_ps(rank + bound_v_i, in_range_m, 
							_mm512_mask_add_ps(_mm512_undefined_ps(), in_range_m, first_term, second_term));
	}

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
	//bot_simd_util.print();

	//_mm_free(n1_buffer);
	//_mm_free(n2_buffer);

}
// End No buffer
////////////////////////////////////////////////////////////

void print(float *rank) 
{
	FILE *fout = fopen("ranks.txt", "w");
	for(unsigned i=0;i<NNODES;i++) {
		//cout << rank[i] << " ";
		fprintf(fout, "%lf\n", rank[i]);
	}
	//cout << endl;
	fclose(fout);
}

void input(char filename[])
{
	//string prefix = string(filename) + "_tiled-" + to_string(TILE_WIDTH);
	string prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-" + to_string(0);
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	fclose(fin);
	unsigned num_tiles;
	unsigned side_length;
	if (NNODES % TILE_WIDTH) {
		side_length = NNODES / TILE_WIDTH + 1;
	} else {
		side_length = NNODES / TILE_WIDTH;
	}
	num_tiles = side_length * side_length;
	// Read the offset and number of edges for every tile
	fname = prefix + "-offsets";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	unsigned *tile_offsets = (unsigned *) malloc(num_tiles * sizeof(unsigned));
	for (unsigned i = 0; i < num_tiles; ++i) {
		fscanf(fin, "%u", tile_offsets + i);
	}
	fclose(fin);
	unsigned *tile_sizes = (unsigned *) malloc(num_tiles * sizeof(unsigned));
	//memset(tile_sizes, 0, num_tiles * sizeof(unsigned));
	for (unsigned i = 0; i < num_tiles; ++i) {
		if (i != num_tiles - 1) {
			tile_sizes[i] = tile_offsets[i + 1] - tile_offsets[i];
		} else {
			tile_sizes[i] = NEDGES - tile_offsets[i];
		}
	}
	// Read graph_degrees
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	unsigned *graph_degrees = (unsigned *) malloc(NNODES * sizeof(unsigned));
	for (unsigned i = 0; i < NNODES; ++i) {
		fscanf(fin, "%u", graph_degrees + i);
	}
	fclose(fin);
	// Read tiles
	unsigned *graph_heads = (unsigned *) _mm_malloc(NEDGES * sizeof(unsigned), ALIGNED_BYTES);
	unsigned *graph_tails = (unsigned *) _mm_malloc(NEDGES * sizeof(unsigned), ALIGNED_BYTES);
	unsigned edge_bound = NEDGES / ALIGNED_BYTES;
	NUM_THREADS = 64;
#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * edge_bound;
	fname = prefix + "-" + to_string(tid);
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	if (0 == tid) {
		fscanf(fin, "%u %u", &NNODES, &NEDGES);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = edge_bound + offset;
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
		//unsigned n1_id = n1 / TILE_WIDTH;
		//unsigned n2_id = n2 / TILE_WIDTH;
		//unsigned tile_id = n1_id * side_length + n2_id;
	}
	fclose(fin);
}

	float *rank = (float *) _mm_malloc(NNODES * sizeof(float), ALIGNED_BYTES);
	float *sum = (float *) _mm_malloc(NNODES * sizeof(float), ALIGNED_BYTES);
	// PageRank
	CHUNK_SIZE = 1;
	//ROW_STEP = 16;
	SIZE_BUFFER_MAX = 512;
	//unsigned ROW_STEP = 128;
	//CHUNK_SIZE = 512;
	//unsigned ROW_STEP = 64;
	NUM_THREADS = 256;
#pragma omp parallel for num_threads(64)
	for (unsigned i = 0; i < NNODES; i++) {
		rank[i] = 1.0;
		sum[i] = 0.0;
	}

	//bot_simd_util.reset();
	for (NUM_THREADS = 64; NUM_THREADS <= 256; NUM_THREADS *= 2) {
		for (int k = 0; k < 10; ++k) {
			page_rank_no_buffer(\
					graph_heads, \
					graph_tails, \
					graph_degrees, \
					tile_sizes, \
					rank, \
					sum, \
					tile_offsets, \
					num_tiles, \
					side_length);
		}
	}

	// Free memory
	_mm_free(graph_heads);
	_mm_free(graph_tails);
	free(graph_degrees);
	free(tile_offsets);
	free(tile_sizes);
	_mm_free(rank);
	_mm_free(sum);
}
int main(int argc, char *argv[]) 
{
	char *filename;
	if (argc > 3) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
		ROW_STEP = strtoul(argv[3], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec_combine/soc-pokec";
		TILE_WIDTH = 1024;
		ROW_STEP = 16;
	}
	input(filename);
	return 0;
}
