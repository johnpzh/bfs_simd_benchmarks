#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits.h>
#include <omp.h>
#include <unistd.h>
#include <getopt.h>
#include <immintrin.h>
#include "../../include/peg_util.h"
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

unsigned NNODES, NEDGES;
unsigned NUM_THREADS; // Number of threads
unsigned TILE_WIDTH; // Width of tile
unsigned SIDE_LENGTH; // Number of rows of tiles
unsigned NUM_TILES; // Number of tiles
unsigned ROW_STEP; // Number of rows of tiles in a Group
unsigned SIZE_BUFFER_MAX;
unsigned T_RATIO;
unsigned WORK_LOAD;
 
double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

////////////////////////////////////////////////////////////
// Weighted Graph version
void input_weighted(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
		unsigned *&graph_weights,
		unsigned *&tile_offsets,
		unsigned *&tile_sizes, 
		unsigned *&graph_vertices,
		unsigned *&graph_edges,
		unsigned *&graph_weights_csr,
		unsigned *&graph_degrees)
{
	//string prefix = string(filename) + "_untiled";
	string file_name_pre = string(filename) + "_weighted_reorder";
	string prefix = file_name_pre + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
	//string prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
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
		fscanf(fin, "%u", tile_offsets + i);
	}
	fclose(fin);
	tile_sizes = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			tile_sizes[i] = tile_offsets[i + 1] - tile_offsets[i];
		} else {
			tile_sizes[i] = NEDGES - tile_offsets[i];
		}
	}

	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_tails = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_weights = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_vertices = (unsigned *) malloc(NNODES * sizeof(unsigned));
	graph_edges = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_weights_csr = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) malloc(NNODES * sizeof(unsigned));

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
		fscanf(fin, "%u%u\n", &NNODES, &NEDGES);
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
		unsigned wt;
		fscanf(fin, "%u%u%u", &n1, &n2, &wt);
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_tails[index] = n2;
		graph_weights[index] = wt;
	}
	fclose(fin);
}
	//For graph CSR
	prefix = file_name_pre + "_untiled";

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
	edge_bound = NEDGES / NUM_THREADS;
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
		fscanf(fin, "%u%u", &NNODES, &NEDGES);
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
		unsigned wt;
		fscanf(fin, "%u%u%u", &n1, &n2, &wt);
		n1--;
		n2--;
		graph_edges[index] = n2;
		graph_weights_csr[index] = wt;
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

////////////////////////////////////////////////////////////
// Dense, Weighted Graph
inline void to_dense(
				unsigned *h_graph_queue,
				const unsigned &frontier_size,
				int *h_graph_mask,
				int *is_active_side)
{
	memset(h_graph_mask, 0, NNODES * sizeof(int));
	memset(is_active_side, 0, SIDE_LENGTH * sizeof(int));

	unsigned remainder = frontier_size % NUM_P_INT;
	unsigned bound_i = frontier_size - remainder;
#pragma omp parallel for
	for (unsigned i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i v_ids_v = _mm512_load_epi32(h_graph_queue + i);
		_mm512_i32scatter_epi32(h_graph_mask, v_ids_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i tw_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(v_ids_v, tw_v);
		_mm512_i32scatter_epi32(is_active_side, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
	if (remainder) {
		__mmask16 in_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512i v_ids_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_m, h_graph_queue + bound_i);
		_mm512_mask_i32scatter_epi32(h_graph_mask, in_m, v_ids_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i tw_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_mask_div_epi32(_mm512_undefined_epi32(), in_m, v_ids_v, tw_v);
		_mm512_mask_i32scatter_epi32(is_active_side, in_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
//#pragma omp parallel for
//	for (unsigned i = 0; i< frontier_size; ++i) {
//		unsigned vertex_id = h_graph_queue[i];
//		h_graph_mask[vertex_id] = 1;
//		is_active_side[vertex_id / TILE_WIDTH] = 1;
//	}
}

inline void update_dense_weighted(
					unsigned &_frontier_size,
					unsigned &_out_degree,
					int *h_graph_mask,
					int *h_updating_graph_mask,
					int *is_active_side,
					int *is_updating_active_side,
					unsigned *graph_degrees)
{
	unsigned frontier_size = 0;
	unsigned out_degree = 0;
#pragma omp parallel for reduction(+: frontier_size, out_degree)
	for (unsigned side_id = 0; side_id < SIDE_LENGTH; ++side_id) {
		if (!is_updating_active_side[side_id]) {
			is_active_side[side_id] = 0;
			unsigned width;
			// think about this bug. how did you find it? and, more importantly,
			// how to avoid it in the future?
			//	memset(h_graph_mask + side_id * TILE_WIDTH, 0, TILE_WIDTH * sizeof(unsigned));
			if (SIDE_LENGTH - 1 != side_id) {
				width = TILE_WIDTH;
			} else {
				width = NNODES - side_id * TILE_WIDTH;
			}
			memset(h_graph_mask + side_id * TILE_WIDTH, 0, width * sizeof(unsigned));
			continue;
		}
		is_updating_active_side[side_id] = 0;
		is_active_side[side_id] = 1;
		unsigned start_vertex_id = side_id * TILE_WIDTH;
		unsigned bound_vertex_id;
		if (SIDE_LENGTH - 1 != side_id) {
			bound_vertex_id = start_vertex_id + TILE_WIDTH;
		} else {
			bound_vertex_id = NNODES;
		}

		unsigned remainder = (bound_vertex_id - start_vertex_id) % NUM_P_INT;
		bound_vertex_id -= remainder;
		for (unsigned vertex_id = start_vertex_id; 
				vertex_id < bound_vertex_id; 
				vertex_id += NUM_P_INT) {
			__m512i updating_flag_v = _mm512_loadu_si512(h_updating_graph_mask + vertex_id);
			__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
			if (!is_updating_m) {
				_mm512_storeu_si512(h_graph_mask + vertex_id, _mm512_set1_epi32(0));// IMPORTANT
				continue;
			}
			_mm512_mask_storeu_epi32(h_updating_graph_mask + vertex_id, is_updating_m, _mm512_set1_epi32(0));
			_mm512_storeu_si512(h_graph_mask + vertex_id, updating_flag_v);
			__m512i num_active_v = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), is_updating_m, 1);
			unsigned num_active = _mm512_reduce_add_epi32(num_active_v);
			frontier_size += num_active;
			__m512i out_degrees_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), is_updating_m, graph_degrees + vertex_id);
			out_degree += _mm512_reduce_add_epi32(out_degrees_v);
		}

		if (remainder > 0) {
			unsigned short in_range_m_t = (unsigned short) 0xFFFF >> (NUM_P_INT - remainder);
			__mmask16 in_range_m = (__mmask16) in_range_m_t;
			__m512i updating_flag_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), in_range_m, h_updating_graph_mask + bound_vertex_id);
			__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
			if (!is_updating_m) {
				_mm512_mask_storeu_epi32(h_graph_mask + bound_vertex_id, in_range_m, _mm512_set1_epi32(0));//addition
				continue;
			}
			_mm512_mask_storeu_epi32(h_updating_graph_mask + bound_vertex_id, is_updating_m, _mm512_set1_epi32(0));
			_mm512_mask_storeu_epi32(h_graph_mask + bound_vertex_id, in_range_m, updating_flag_v);
			__m512i num_active_v = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), is_updating_m, 1);
			unsigned num_active = _mm512_reduce_add_epi32(num_active_v);
			frontier_size += num_active;
			__m512i out_degrees_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), is_updating_m, graph_degrees + bound_vertex_id);
			out_degree += _mm512_reduce_add_epi32(out_degrees_v);
		}

		//for (unsigned vertex_id = start_vertex_id; vertex_id < bound_vertex_id; ++vertex_id) {
		//	if (1 == h_updating_graph_mask[vertex_id]) {
		//		h_updating_graph_mask[vertex_id] = 0;
		//		h_graph_mask[vertex_id] = 1;
		//		frontier_size++;
		//		out_degree += graph_degrees[vertex_id];
		//	} else {
		//		h_graph_mask[vertex_id] = 0;
		//	}
		//}
	}

	_frontier_size = frontier_size;
	_out_degree = out_degree;
}

// Scan the data, accumulate the values with the same index.
// Then, store the cumulative sum to the last element in the data with the same index.
inline void mask_min_conflict_safe_epi32(
									__mmask16 valid_m,
									__m512i &data,
									__m512i indices)
{
	//__m512i cd = _mm512_conflict_epi32(indices);
	__m512i cd = _mm512_mask_conflict_epi32(_mm512_set1_epi32(0), valid_m, indices);
	__mmask16 todo_mask = _mm512_test_epi32_mask(cd, _mm512_set1_epi32(-1));
	if (todo_mask) {
		__m512i lz = _mm512_lzcnt_epi32(cd);
		__m512i lid = _mm512_sub_epi32(_mm512_set1_epi32(31), lz);
		while (todo_mask) {
			__m512i todo_bcast = _mm512_broadcastmw_epi32(todo_mask);
			__mmask16 now_mask = _mm512_mask_testn_epi32_mask(todo_mask, cd, todo_bcast);
			__m512i data_perm = _mm512_mask_permutexvar_epi32(_mm512_undefined_epi32(), now_mask, lid, data);
			__mmask16 shorter_m = _mm512_mask_cmplt_epi32_mask(now_mask, data_perm, data);
			data = _mm512_mask_mov_epi32(data, shorter_m, data_perm);
			//data = _mm512_mask_add_epi32(data, now_mask, data, data_perm);
			todo_mask = _mm512_kxor(todo_mask, now_mask);
		}
	} 
}

inline void sssp_kernel_dense_weighted(
				unsigned *heads_buffer,
				unsigned *tails_buffer,
				unsigned *weights_buffer,
				const unsigned &size_buffer,
				int *h_graph_mask, 
				int *h_updating_graph_mask,
				int *is_updating_active_side,
				unsigned *dists) 
{
	unsigned remainder = size_buffer % NUM_P_INT;
	unsigned bound_edge_i = size_buffer - remainder;
	for (unsigned edge_i = 0; edge_i < bound_edge_i; edge_i += NUM_P_INT) {
		__m512i head_v = _mm512_load_epi32(heads_buffer + edge_i);
		__m512i active_flag_v = _mm512_i32gather_epi32(head_v, h_graph_mask, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			continue;
		}
		__m512i end_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, tails_buffer + edge_i);
		__m512i dists_end_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), is_active_m, end_v, dists, sizeof(unsigned));
		__m512i dists_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), is_active_m, head_v, dists, sizeof(unsigned));
		__m512i weights_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, weights_buffer + edge_i);
		__m512i dists_tmp_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), is_active_m, dists_head_v, weights_v);
		//__mmask16 is_minusone_m = _mm512_mask_cmpeq_epi32_mask(is_active_m, dists_end_v, _mm512_set1_epi32(-1));
		__mmask16 need_update_m = _mm512_mask_cmplt_epi32_mask(is_active_m, dists_tmp_v, dists_end_v);
		//__mmask16 is_shorter_m = _mm512_mask_cmplt_epi32_mask(is_active_m, dists_tmp_v, dists_end_v);
		//__mmask16 need_update_m = is_minusone_m | is_shorter_m;
		//__mmask16 need_update_m = is_shorter_m;
		if (!need_update_m) {
			continue;
		}
		//if (!is_shorter_m) {
		//	continue;
		//}
		dists_tmp_v = _mm512_mask_mov_epi32(_mm512_set1_epi32(INT_MAX), need_update_m, dists_tmp_v);
		mask_min_conflict_safe_epi32(need_update_m, dists_tmp_v, end_v);
		_mm512_mask_i32scatter_epi32(dists, need_update_m, end_v, dists_tmp_v, sizeof(unsigned));

		__m512i updating_active_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), need_update_m, end_v, h_updating_graph_mask, sizeof(int));
		__mmask16 not_updating_active_m = _mm512_testn_epi32_mask(updating_active_v, _mm512_set1_epi32(-1));
		if (!not_updating_active_m) {
			continue;
		}
		_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_updating_active_m, end_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(end_v, TILE_WIDTH_v);
		_mm512_mask_i32scatter_epi32(is_updating_active_side, not_updating_active_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}

	if (remainder > 0) {
		__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512i head_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, heads_buffer + bound_edge_i);
		__m512i active_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, head_v, h_graph_mask, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			return;
		}
		__m512i end_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, tails_buffer + bound_edge_i);
		__m512i dists_end_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), is_active_m, end_v, dists, sizeof(unsigned));
		__m512i dists_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), is_active_m, head_v, dists, sizeof(unsigned));
		__m512i weights_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, weights_buffer + bound_edge_i);
		__m512i dists_tmp_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), is_active_m, dists_head_v, weights_v);
		//__mmask16 is_minusone_m = _mm512_mask_cmpeq_epi32_mask(is_active_m, dists_end_v, _mm512_set1_epi32(-1));
		__mmask16 need_update_m = _mm512_mask_cmplt_epi32_mask(is_active_m, dists_tmp_v, dists_end_v);
		//__mmask16 is_shorter_m = _mm512_mask_cmplt_epi32_mask(is_active_m, dists_tmp_v, dists_end_v);
		//__mmask16 need_update_m = is_minusone_m | is_shorter_m;
		//__mmask16 need_update_m = is_shorter_m;
		if (!need_update_m) {
			return;
		}
		//if (!is_shorter_m) {
		//	return;
		//}
		dists_tmp_v = _mm512_mask_mov_epi32(_mm512_set1_epi32(INT_MAX), need_update_m, dists_tmp_v);
		mask_min_conflict_safe_epi32(need_update_m, dists_tmp_v, end_v);
		_mm512_mask_i32scatter_epi32(dists, need_update_m, end_v, dists_tmp_v, sizeof(unsigned));

		__m512i updating_active_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), need_update_m, end_v, h_updating_graph_mask, sizeof(int));
		__mmask16 not_updating_active_m = _mm512_testn_epi32_mask(updating_active_v, _mm512_set1_epi32(-1));
		if (!not_updating_active_m) {
			return;
		}
		_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_updating_active_m, end_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(end_v, TILE_WIDTH_v);
		_mm512_mask_i32scatter_epi32(is_updating_active_side, not_updating_active_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
//	for (unsigned edge_i = 0; edge_i < size_buffer; ++edge_i) {
//		unsigned head = heads_buffer[edge_i];
//		if (0 == h_graph_mask[head]) {
//			continue;
//		}
//		unsigned end = tails_buffer[edge_i];
//		unsigned new_dist = dists[head] + weights_buffer[edge_i];
//		if (new_dist < dists[end]) {
//			dists[end] = new_dist;
//			h_updating_graph_mask[end] = 1;
//			is_updating_active_side[end/TILE_WIDTH] = 1;
//		}
//	}
}

inline void scheduler_dense_weighted(
					unsigned *heads_buffer,
					unsigned *tails_buffer,
					unsigned *weights_buffer,
					unsigned *graph_heads, 
					unsigned *graph_tails, 
					unsigned *graph_weights,
					unsigned *tile_offsets,
					unsigned *tile_sizes,
					int *h_graph_mask, 
					int *h_updating_graph_mask,
					int *is_active_side,
					int *is_updating_active_side,
					unsigned *dists, 
					const unsigned &start_row_index,
					const unsigned &tile_step)
{
	unsigned start_tile_id = start_row_index * SIDE_LENGTH;
	//unsigned bound_row_id = start_row_index + tile_step;
	unsigned end_tile_id = start_tile_id + tile_step * SIDE_LENGTH;
#pragma omp parallel for schedule(dynamic, 1)
//#pragma omp parallel for
	for (unsigned tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
		unsigned bound_tile_id = tile_index + tile_step;
		unsigned tid = omp_get_thread_num();
		unsigned *heads_buffer_base = heads_buffer + tid * SIZE_BUFFER_MAX;
		unsigned *tails_buffer_base = tails_buffer + tid * SIZE_BUFFER_MAX;
		unsigned *weights_buffer_base = weights_buffer + tid * SIZE_BUFFER_MAX;
		unsigned size_buffer = 0;
		unsigned capacity = SIZE_BUFFER_MAX;
		for (unsigned tile_id = tile_index; tile_id < bound_tile_id; ++tile_id) {
			unsigned row_id = (tile_id - start_tile_id) % tile_step + start_row_index;
			if (0 == tile_sizes[tile_id] || !is_active_side[row_id]) {
				continue;
			}
			// Load to buffer
			unsigned edge_i = tile_offsets[tile_id];
			unsigned remain = tile_sizes[tile_id];
			while (remain != 0) {
				if (capacity > 0) {
					if (capacity > remain) {
						// Put all remain into the buffer
						memcpy(heads_buffer_base + size_buffer, graph_heads + edge_i, remain * sizeof(unsigned));
						memcpy(tails_buffer_base + size_buffer, graph_tails + edge_i, remain * sizeof(unsigned));
						memcpy(weights_buffer_base + size_buffer, graph_weights + edge_i, remain * sizeof(unsigned));
						edge_i += remain;
						capacity -= remain;
						size_buffer += remain;
						remain = 0;
					} else {
						// Fill the buffer to full
						memcpy(heads_buffer_base + size_buffer, graph_heads + edge_i, capacity * sizeof(unsigned));
						memcpy(tails_buffer_base + size_buffer, graph_tails + edge_i, capacity * sizeof(unsigned));
						memcpy(weights_buffer_base + size_buffer, graph_weights + edge_i, capacity * sizeof(unsigned));
						edge_i += capacity;
						remain -= capacity;
						size_buffer += capacity;
						capacity = 0;
					}
				} else {
					// Process the full buffer
					sssp_kernel_dense_weighted(
							heads_buffer_base,
							tails_buffer_base,
							weights_buffer_base,
							size_buffer,
							h_graph_mask, 
							h_updating_graph_mask,
							is_updating_active_side,
							dists); 
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}
			
		}
		// Process the remains in buffer
		sssp_kernel_dense_weighted(
				heads_buffer_base,
				tails_buffer_base,
				weights_buffer_base,
				size_buffer,
				h_graph_mask, 
				h_updating_graph_mask,
				is_updating_active_side,
				dists); 
			//sssp_kernel_dense_weighted(
			//	graph_heads, 
			//	graph_tails, 
			//	graph_weights,
			//	h_graph_mask, 
			//	h_updating_graph_mask,
			//	is_active_side,
			//	is_updating_active_side,
			//	dists, 
			//	tile_offsets[tile_id], 
			//	bound_edge_i);
	}
}
inline void BFS_dense_weighted(
					unsigned *graph_heads,
					unsigned *graph_tails,
					unsigned *graph_weights,
					unsigned *tile_offsets,
					unsigned *tile_sizes,
					int *h_graph_mask,
					int *h_updating_graph_mask,
					int *is_active_side,
					int *is_updating_active_side,
					unsigned *dists)
{
	unsigned *heads_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned *tails_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned *weights_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned remainder = SIDE_LENGTH % ROW_STEP;
	unsigned bound_side_id = SIDE_LENGTH - remainder;
	for (unsigned side_id = 0; side_id < bound_side_id; side_id += ROW_STEP) {
		scheduler_dense_weighted(
				heads_buffer,
				tails_buffer,
				weights_buffer,
				graph_heads, 
				graph_tails, 
				graph_weights,
				tile_offsets,
				tile_sizes,
				h_graph_mask, 
				h_updating_graph_mask,
				is_active_side,
				is_updating_active_side,
				dists, 
				side_id,
				ROW_STEP);
		//side_id += ROW_STEP;
	}
	if (remainder > 0) {
		scheduler_dense_weighted(
				heads_buffer,
				tails_buffer,
				weights_buffer,
				graph_heads, 
				graph_tails, 
				graph_weights,
				tile_offsets,
				tile_sizes,
				h_graph_mask, 
				h_updating_graph_mask,
				is_active_side,
				is_updating_active_side,
				dists, 
				bound_side_id,
				remainder);
	}
	_mm_free(heads_buffer);
	_mm_free(tails_buffer);
	_mm_free(weights_buffer);


}
// end dense, weighted graph
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// sparse, weighted graph

inline unsigned get_chunk_size(unsigned amount)
{
	unsigned r = amount / NUM_THREADS / WORK_LOAD;
	if (r) {
		return r;
	} else {
		return 1;
	}
}
inline unsigned *to_sparse(
		int *h_graph_mask,
		const unsigned &frontier_size)
{
	//unsigned *new_frontier = (unsigned *) malloc(frontier_size * sizeof(unsigned));
	unsigned *new_frontier = (unsigned *) _mm_malloc(frontier_size * sizeof(unsigned), ALIGNED_BYTES);
	const unsigned block_size = 1 << 12;
	unsigned num_blocks = (NNODES - 1)/block_size + 1;
	unsigned *nums_in_blocks = nullptr;
	
	if (num_blocks > 1) {
		nums_in_blocks = (unsigned *) malloc(num_blocks * sizeof(unsigned));
		memset(nums_in_blocks, 0, num_blocks * sizeof(unsigned));
		// The start locations where the vertices are put in the frontier.
#pragma omp parallel for
		for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
			unsigned offset = block_i * block_size;
			unsigned bound;
			if (num_blocks - 1 != block_i) {
				bound = offset + block_size;
			} else {
				bound = NNODES;
			}
			for (unsigned vertex_i = offset; vertex_i < bound; ++vertex_i) {
				if (h_graph_mask[vertex_i]) {
					nums_in_blocks[block_i]++;
				}
			}
		}
		//TODO: blocked parallel for
		// Scan to get the offsets as start locations.
		unsigned offset_sum = 0;
		for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
			unsigned tmp = nums_in_blocks[block_i];
			nums_in_blocks[block_i] = offset_sum;
			offset_sum += tmp;
		}
		// Put vertices into the frontier
#pragma omp parallel for
		for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
			unsigned base = nums_in_blocks[block_i];
			unsigned offset = block_i * block_size;
			unsigned bound;
			if (num_blocks - 1 != block_i) {
				bound = offset + block_size;
			} else {
				bound = NNODES;
			}
			for (unsigned vertex_i = offset; vertex_i < bound; ++vertex_i) {
				if (h_graph_mask[vertex_i]) {
					new_frontier[base++] = vertex_i;
				}
			}
		}
		free(nums_in_blocks);
	} else {
		unsigned k = 0;
		for (unsigned i = 0; i < NNODES; ++i) {
			if (h_graph_mask[i]) {
				new_frontier[k++] = i;
			}
		}
	}
	return new_frontier;
}

inline unsigned update_sparse_weighted(
				unsigned *h_graph_queue,
				const unsigned &queue_size,
				unsigned *graph_degrees,
				int *h_updating_graph_mask)
{
//	unsigned out_degree = 0;
//#pragma omp parallel for reduction(+: out_degree)
//	for (unsigned i = 0; i < queue_size; ++i) {
//		unsigned vertex_id = h_graph_queue[i];
//		out_degree += graph_degrees[vertex_id];
//		visited[vertex_id] = 0;
//	}
	unsigned out_degree = 0;
	unsigned remainder = queue_size % NUM_P_INT;
	unsigned bound_i = queue_size - remainder;
#pragma omp parallel for reduction(+: out_degree)
	for (unsigned i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i vertex_id_v = _mm512_load_epi32(h_graph_queue + i);
		__m512i degrees_v = _mm512_i32gather_epi32(vertex_id_v, graph_degrees, sizeof(unsigned));
		unsigned sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		out_degree += sum_degrees;
		_mm512_i32scatter_epi32(h_updating_graph_mask, vertex_id_v, _mm512_set1_epi32(0), sizeof(int));
	}
	if (remainder) {
		__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xffff >> (NUM_P_INT - remainder));
		__m512i vertex_id_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, h_graph_queue + bound_i);
		__m512i degrees_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, vertex_id_v, graph_degrees, sizeof(unsigned));
		unsigned sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		out_degree += sum_degrees;
		_mm512_mask_i32scatter_epi32(h_updating_graph_mask, in_range_m, vertex_id_v, _mm512_set1_epi32(0), sizeof(int));
	}
	return out_degree;
}

inline unsigned *BFS_kernel_sparse_weighted(
				unsigned *graph_vertices,
				unsigned *graph_edges,
				unsigned *graph_weights_csr,
				unsigned *graph_degrees,
				//int *h_graph_visited,
				unsigned *h_graph_queue,
				unsigned &queue_size,
				//unsigned *num_paths,
				int *visited,
				unsigned *dists)
{
	//int *visited = (int *) calloc(NNODES, sizeof(int));
	// from h_graph_queue, get the degrees (para_for)
	unsigned *degrees = (unsigned *) _mm_malloc(sizeof(unsigned) *  queue_size, ALIGNED_BYTES);
	unsigned new_queue_size = 0;
	unsigned remainder = queue_size % NUM_P_INT;
	unsigned bound_i = queue_size - remainder;
//#pragma omp parallel for schedule(dynamic) reduction(+: new_queue_size)
#pragma omp parallel for reduction(+: new_queue_size)
	for (unsigned i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i v_ids = _mm512_load_epi32(h_graph_queue + i);
		__m512i degrees_v = _mm512_i32gather_epi32(v_ids, graph_degrees, sizeof(unsigned));
		_mm512_store_epi32(degrees + i, degrees_v);
		unsigned sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		new_queue_size += sum_degrees;
	}
	if (remainder) {
		__mmask16 in_m = (__mmask16) ((unsigned short) 0xffff >> (NUM_P_INT - remainder));
		__m512i v_ids = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_m, h_graph_queue + bound_i);
		__m512i degrees_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_m, v_ids, graph_degrees, sizeof(unsigned));
		_mm512_mask_store_epi32(degrees + bound_i, in_m, degrees_v);
		unsigned sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		new_queue_size += sum_degrees;
	}
//#pragma omp parallel for reduction(+: new_queue_size)
//	for (unsigned i = 0; i < queue_size; ++i) {
//		degrees[i] = graph_degrees[h_graph_queue[i]];
//		new_queue_size += degrees[i];
//	}
	if (0 == new_queue_size) {
		_mm_free(degrees);
		queue_size = 0;
		return nullptr;
	}

	// from degrees, get the offset (stored in degrees) (block_para_for)
	// todo: blocked parallel for
	unsigned offset_sum = 0;
	for (unsigned i = 0; i < queue_size; ++i) {
		unsigned tmp = degrees[i];
		degrees[i] = offset_sum;
		offset_sum += tmp;
	}

	// from offset, get active vertices (para_for)
	unsigned *new_frontier_tmp = (unsigned *) _mm_malloc(sizeof(unsigned) * new_queue_size, ALIGNED_BYTES);
	unsigned chunk_size_sparse = get_chunk_size(queue_size);
#pragma omp parallel for schedule(dynamic, chunk_size_sparse)
	for (unsigned i = 0; i < queue_size; ++i) {
		unsigned start = h_graph_queue[i];
		unsigned offset = degrees[i];
		unsigned out_degree = graph_degrees[start];
		unsigned base = graph_vertices[start];
		for (unsigned k = 0; k < out_degree; ++k) {
			unsigned frontier_i = offset + k;
			unsigned edge_i = base + k;
			unsigned end = graph_edges[edge_i];
			unsigned new_dist = dists[start] + graph_weights_csr[edge_i];

			bool dist_written = false;
			volatile unsigned old_val;
			volatile unsigned new_val;
			do {
				old_val = dists[end];
				new_val = new_dist;
			} while (old_val > new_val 
					&& !(dist_written = __sync_bool_compare_and_swap(dists + end, old_val, new_val)));
			//if (dist_written && __sync_bool_compare_and_swap(visited + end, 0, 1)) {
			//	new_frontier_tmp[frontier_i] = end;
			//} else {
			//	new_frontier_tmp[frontier_i] = (unsigned) -1;
			//}
			if (dist_written && !visited[end]) {
				visited[end] = 1;
				new_frontier_tmp[frontier_i] = end;
			} else {
				new_frontier_tmp[frontier_i] = (unsigned) -1;
			}

			//if (new_dist < dists[end]) {
			//	volatile unsigned old_val = dists[end];
			//	volatile unsigned new_val = new_dist;
			//	bool dist_updated = __sync_bool_compare_and_swap(dists + end, old_val, new_val);
			//	if (dist_updated) {
			//		new_frontier_tmp[frontier_i] = end;
			//	} else {
			//		new_frontier_tmp[frontier_i] = (unsigned) -1;
			//	}
			//} else {
			//	new_frontier_tmp[frontier_i] = (unsigned) -1;
			//}
		}
	}

	//free(visited);

	// refine active vertices, removing visited and redundant (block_para_for)
	unsigned block_size = 1024 * 2;
	unsigned num_blocks = (new_queue_size - 1)/block_size + 1;

	unsigned *nums_in_blocks = nullptr;
	if (num_blocks > 1) {
	nums_in_blocks = (unsigned *) malloc(sizeof(unsigned) * num_blocks);
	unsigned new_queue_size_tmp = 0;
//#pragma omp parallel for schedule(dynamic) reduction(+: new_queue_size_tmp)
#pragma omp parallel for reduction(+: new_queue_size_tmp)
	for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
		unsigned offset = block_i * block_size;
		unsigned bound;
		if (num_blocks - 1 != block_i) {
			bound = offset + block_size;
		} else {
			bound = new_queue_size;
		}
		unsigned base = offset;
		for (unsigned end_i = offset; end_i < bound; ++end_i) {
			if ((unsigned) - 1 != new_frontier_tmp[end_i]) {
				new_frontier_tmp[base++] = new_frontier_tmp[end_i];
			}
		}
		nums_in_blocks[block_i] = base - offset;
		new_queue_size_tmp += nums_in_blocks[block_i];
	}
	new_queue_size = new_queue_size_tmp;
	} else {
		unsigned base = 0;
		for (unsigned i = 0; i < new_queue_size; ++i) {
			if ((unsigned) -1 != new_frontier_tmp[i]) {
				new_frontier_tmp[base++] = new_frontier_tmp[i];
			}
		}
		new_queue_size = base;
	}
	
	if (0 == new_queue_size) {
		_mm_free(degrees);
		_mm_free(new_frontier_tmp);
		if (nums_in_blocks) {
			free(nums_in_blocks);
		}
		queue_size = 0;
		return nullptr;
	}

	// get the final new frontier
	unsigned *new_frontier = (unsigned *) _mm_malloc(sizeof(unsigned) * new_queue_size, ALIGNED_BYTES);
	//unsigned *new_frontier = (unsigned *) malloc(sizeof(unsigned) * new_queue_size);
	if (num_blocks > 1) {
		//todo: blocked parallel for
		offset_sum = 0;
		for (unsigned i = 0; i < num_blocks; ++i) {
			unsigned tmp = nums_in_blocks[i];
			nums_in_blocks[i] = offset_sum;
			offset_sum += tmp;
		}
		//#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for
		for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
			unsigned offset = nums_in_blocks[block_i];
			unsigned bound;
			if (num_blocks - 1 != block_i) {
				bound = nums_in_blocks[block_i + 1];
			} else {
				bound = new_queue_size;
			}
			unsigned base = block_i * block_size;
			unsigned remainder = (bound - offset) % NUM_P_INT;
			unsigned bound_i = bound - remainder;
			for (unsigned i = offset; i < bound_i; i += NUM_P_INT) {
				__m512i tmp = _mm512_load_epi32(new_frontier_tmp + base);
				_mm512_storeu_si512(new_frontier + i, tmp);
				base += NUM_P_INT;
			}
			if (remainder) {
				__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xffff >> (NUM_P_INT - remainder));
				__m512i tmp = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, new_frontier_tmp + base);
				_mm512_mask_storeu_epi32(new_frontier + bound_i, in_range_m, tmp);
			}
			//for (unsigned i = offset; i < bound; ++i) {
			//	new_frontier[i] = new_frontier_tmp[base++];
			//}
		}
	} else {
		unsigned remainder = new_queue_size % NUM_P_INT;
		unsigned bound_i = new_queue_size - remainder;
		for (unsigned i = 0; i < bound_i; i += NUM_P_INT) {
			__m512i tmp = _mm512_load_epi32(new_frontier_tmp + i);
			_mm512_store_epi32(new_frontier + i, tmp);
		}
		if (remainder) {
			__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xffff >> (NUM_P_INT - remainder));
			__m512i tmp = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, new_frontier_tmp + bound_i);
			_mm512_mask_store_epi32(new_frontier + bound_i, in_range_m, tmp);
		}
		//unsigned base = 0;
		//for (unsigned i = 0; i < new_queue_size; ++i) {
		//	new_frontier[i] = new_frontier_tmp[base++];
		//}
	}

	// return the results
	_mm_free(degrees);
	_mm_free(new_frontier_tmp);
	if (nums_in_blocks) {
		free(nums_in_blocks);
	}
	queue_size = new_queue_size;
	return new_frontier;
}
inline unsigned *BFS_sparse_weighted(
				unsigned *h_graph_queue,
				unsigned &queue_size,
				unsigned *graph_vertices,
				unsigned *graph_edges,
				unsigned *graph_degrees,
				unsigned *graph_weights_csr,
				//int *h_graph_visited,
				//unsigned *num_paths
				int *h_updating_graph_mask,
				unsigned *dists)
{
	return BFS_kernel_sparse_weighted(
				graph_vertices,
				graph_edges,
				graph_weights_csr,
				graph_degrees,
				//h_graph_visited,
				h_graph_queue,
				queue_size,
				//num_paths,
				h_updating_graph_mask,
				dists);
}
// end sparse, weighted graph
////////////////////////////////////////////////////////////

void sssp_weighted(
		unsigned *graph_heads, 
		unsigned *graph_tails, 
		unsigned *graph_weights,
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		unsigned *graph_vertices,
		unsigned *graph_edges,
		unsigned *graph_weights_csr,
		unsigned *graph_degrees,
		const unsigned &source)
{
	omp_set_num_threads(NUM_THREADS);

	unsigned *dists = (unsigned *) malloc(NNODES * sizeof(int));
	int *h_graph_mask = (int *) malloc(NNODES * sizeof(int));
	int *h_updating_graph_mask = (int *) malloc(NNODES * sizeof(int));
	int *new_mask = nullptr;
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned *h_graph_queue = nullptr;
	unsigned frontier_size;
	unsigned out_degree;
	bool last_is_dense = true;

	//memset(dists, -1, NNODES * sizeof(int));
	for (unsigned i = 0; i < NNODES; ++i) {
		dists[i] = INT_MAX;
	}
	dists[source] = 0;

	double start_time = omp_get_wtime();

	// Fisrt is Sparse
	frontier_size = 1;
	h_graph_queue = (unsigned *) _mm_malloc(frontier_size *sizeof(unsigned), ALIGNED_BYTES);
	h_graph_queue[0] = source;
	unsigned *new_queue = BFS_sparse_weighted(
								h_graph_queue,
								frontier_size,
								graph_vertices,
								graph_edges,
								graph_degrees,
								graph_weights_csr,
								h_updating_graph_mask,
								dists);
	_mm_free(h_graph_queue); 
	h_graph_queue = new_queue;
	last_is_dense = false;
	// Get the sum of the number of active nodes and their out degrees
	out_degree =  update_sparse_weighted(
							h_graph_queue,
							frontier_size,
							graph_degrees,
							h_updating_graph_mask);
	h_graph_mask[source] = 1;
	is_active_side[source/TILE_WIDTH] = 1;

	unsigned pattern_threshold = NEDGES / T_RATIO;

	while (true) {
		if (frontier_size + out_degree > pattern_threshold) {
			// Dense
			if (!last_is_dense) {
				to_dense(
					h_graph_queue,
					frontier_size,
					h_graph_mask,
					is_active_side);
			}
			BFS_dense_weighted(
					graph_heads,
					graph_tails,
					graph_weights,
					tile_offsets,
					tile_sizes,
					h_graph_mask,
					h_updating_graph_mask,
					is_active_side,
					is_updating_active_side,
					dists);
			last_is_dense = true;
			update_dense_weighted(
					frontier_size,
					out_degree,
					h_graph_mask,
					h_updating_graph_mask,
					is_active_side,
					is_updating_active_side,
					graph_degrees);
			if (0 == frontier_size) {
				break;
			}
		} else {
			// Sparse
			if (last_is_dense) {
				new_queue = to_sparse(
							h_graph_mask,
							frontier_size);
				_mm_free(h_graph_queue);
				h_graph_queue = new_queue;
			}
			new_queue = BFS_sparse_weighted(
									h_graph_queue,
									frontier_size,
									graph_vertices,
									graph_edges,
									graph_degrees,
									graph_weights_csr,
									h_updating_graph_mask,
									dists);
			_mm_free(h_graph_queue);
			h_graph_queue = new_queue;
			last_is_dense = false;
			if (0 == frontier_size) {
				break;
			}
			out_degree =  update_sparse_weighted(
					h_graph_queue,
					frontier_size,
					graph_degrees,
					h_updating_graph_mask);
		}
	}

	double end_time = omp_get_wtime();
	double run_time;
	printf("%u %lf\n", NUM_THREADS, run_time = end_time - start_time);
	bot_best_perform.record(run_time, NUM_THREADS);

	////test
	//FILE *fout = fopen("results.txt", "w");
	//for (unsigned i = 0; i < NNODES; ++i) {
	//	fprintf(fout, "%u: %u\n", i, dists[i]);
	//}
	//fclose(fout);

	free(dists);
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(is_active_side);
	free(is_updating_active_side);
	_mm_free(h_graph_queue);
}
// End Weighted Graph
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// Unweighted Graph (DEPRECATED!)
void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
		unsigned *&tile_offsets,
		unsigned *&tile_sizes) 
{
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
		fscanf(fin, "%u", tile_offsets + i);
	}
	fclose(fin);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_tails = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	tile_sizes = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			tile_sizes[i] = tile_offsets[i + 1] - tile_offsets[i];
		} else {
			tile_sizes[i] = NEDGES - tile_offsets[i];
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
		graph_tails[index] = n2;
	}
	fclose(fin);
}
}

void print(int *dists) {
	FILE *fout = fopen("distances.txt", "w");
	for(unsigned i=0;i<NNODES;i++) {
		fprintf(fout, "%d\n", dists[i]);
	}
	fclose(fout);
}

inline void sssp_kernel(
				unsigned *graph_heads, 
				unsigned *graph_tails, 
				int *h_graph_mask, 
				int *h_updating_graph_mask,
				int *is_active_side,
				int *is_updating_active_side,
				int *dists, 
				const unsigned &edge_i_start, 
				const unsigned &edge_i_bound)
{
	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		if (0 == h_graph_mask[head]) {
			continue;
		}
		unsigned end = graph_tails[edge_i];
		if (-1 == dists[end] || dists[head] + 1 < dists[end]) {
			dists[end] = dists[head] + 1;
			h_updating_graph_mask[end] = 1;
			is_updating_active_side[end/TILE_WIDTH] = 1;
		}
	}
}

inline void scheduler(
					unsigned *graph_heads, 
					unsigned *graph_tails, 
					unsigned *tile_offsets,
					int *h_graph_mask, 
					int *h_updating_graph_mask,
					int *is_active_side,
					int *is_updating_active_side,
					unsigned *tile_sizes,
					int *dists, 
					const unsigned &start_row_index,
					const unsigned &tile_step)
{
	unsigned bound_row_index = start_row_index + tile_step;
//#pragma omp parallel for schedule(dynamic, 1)
#pragma omp parallel for
	for (unsigned col_id = 0; col_id < SIDE_LENGTH; ++col_id) {
		for (unsigned row_id = start_row_index; row_id < bound_row_index; ++row_id) {
			unsigned tile_id = row_id * SIDE_LENGTH + col_id;
			if (0 == tile_sizes[tile_id] || !is_active_side[row_id]) {
				continue;
			}
			//bfs_kernel();
			unsigned bound_edge_i;
			if (NUM_TILES - 1 != tile_id) {
				bound_edge_i = tile_offsets[tile_id + 1];
			} else {
				bound_edge_i = NEDGES;
			}
			sssp_kernel(
				graph_heads, 
				graph_tails, 
				h_graph_mask, 
				h_updating_graph_mask,
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
		unsigned *graph_tails, 
		unsigned *tile_offsets,
		//int *h_graph_mask, 
		//int *h_updating_graph_mask,
		//int *is_active_side,
		//int *is_updating_active_side,
		unsigned *tile_sizes,
		//int *dists,
		const unsigned source)
{
	omp_set_num_threads(NUM_THREADS);

	int *dists = (int *) malloc(NNODES * sizeof(int));
	int *h_graph_mask = (int *) malloc(NNODES * sizeof(int));
	int *h_updating_graph_mask = (int *) malloc(NNODES * sizeof(int));
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);

	memset(dists, -1, NNODES * sizeof(int));
	dists[source] = 0;
	memset(h_graph_mask, 0, NNODES * sizeof(int));
	h_graph_mask[source] = 1;
	memset(h_updating_graph_mask, 0, NNODES * sizeof(int));
	memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
	is_active_side[source/TILE_WIDTH] = 1;
	memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);

	double start_time = omp_get_wtime();
	int stop = 0;
	while (!stop) {
		stop = 1;
		unsigned remainder = SIDE_LENGTH % ROW_STEP;
		unsigned bound_side_id = SIDE_LENGTH - remainder;
		for (unsigned side_id = 0; side_id < bound_side_id; side_id += ROW_STEP) {
			//if (!is_active_side[side_id]) {
			//	++side_id;
			//	continue;
			//}
			scheduler(
				graph_heads, 
				graph_tails, 
				tile_offsets,
				h_graph_mask, 
				h_updating_graph_mask,
				is_active_side,
				is_updating_active_side,
				tile_sizes,
				dists, 
				side_id,
				ROW_STEP);
			//side_id += ROW_STEP;
		}
		if (remainder > 0) {
			scheduler(
				graph_heads, 
				graph_tails, 
				tile_offsets,
				h_graph_mask, 
				h_updating_graph_mask,
				is_active_side,
				is_updating_active_side,
				tile_sizes,
				dists, 
				bound_side_id,
				remainder);
		}
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
				if (1 == h_updating_graph_mask[vertex_id]) {
					h_updating_graph_mask[vertex_id] = 0;
					h_graph_mask[vertex_id] = 1;
				} else {
					h_graph_mask[vertex_id] = 0;
				}
			}
		}
	}

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);

	free(dists);
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(is_active_side);
	free(is_updating_active_side);
}
// End Unweighted Graph
////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) 
{
	int is_weighted_graph = 0;
	// Process the options
	char *filename;
	if (argc > 3) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
		ROW_STEP = strtoul(argv[3], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/coo_tiled_bak/soc-pokec";
		TILE_WIDTH = 1024;
		ROW_STEP = 16;
	}

	int arg_flag;
	while (1) {
		static option long_options[] = {
			{"weighted", no_argument, 0, 'w'},
			{0, 0, 0, 0}
		};
		int option_index = 0;
		arg_flag = getopt_long (argc, argv, "w", long_options, &option_index);

		if (-1 == arg_flag) {
			break;
		}

		switch (arg_flag) {
			case 'w':
				is_weighted_graph = 1;
				break;
			default:
				// Need to do something here if all option process has been combined here.
				break;
		}
	}
	// End Process the options
	
	// Input
	unsigned *graph_heads;
	unsigned *graph_tails;
	unsigned *graph_weights = nullptr;
	unsigned *tile_offsets;
	unsigned *tile_sizes;

	unsigned *graph_vertices = nullptr;
	unsigned *graph_edges = nullptr;
	unsigned *graph_weights_csr = nullptr;
	unsigned *graph_degrees = nullptr;
	//unsigned *nneibor;
	if (is_weighted_graph) {
		input_weighted(
				filename, 
				graph_heads, 
				graph_tails, 
				graph_weights,
				tile_offsets,
				tile_sizes,
				graph_vertices,
				graph_edges,
				graph_weights_csr,
				graph_degrees);
	} else {
		input(
				filename, 
				graph_heads, 
				graph_tails, 
				tile_offsets,
				tile_sizes);
	}

	// SSSP
	//int *distances = (int *) malloc(NNODES * sizeof(int));
	//int *h_graph_mask = (int *) malloc(NNODES * sizeof(int));
	//int *h_updating_graph_mask = (int *) malloc(NNODES * sizeof(int));
	//int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	//int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned source = 0;
	
#ifdef ONEDEBUG
	printf("SSSP starts...\n");
	unsigned run_count = 9;
#else
	unsigned run_count = 9;
#endif
	T_RATIO = 20;
	WORK_LOAD = 30;
	SIZE_BUFFER_MAX = 512;
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		bot_best_perform.reset();
		//memset(distances, -1, NNODES * sizeof(int));
		//distances[source] = 0;
		//memset(h_graph_mask, 0, NNODES * sizeof(int));
		//h_graph_mask[source] = 1;
		//memset(h_updating_graph_mask, 0, NNODES * sizeof(int));
		//memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
		//is_active_side[source/TILE_WIDTH] = 1;
		//memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);

		//sleep(10);
		for (int k = 0; k < 10; ++k) {
		if (is_weighted_graph) {
			sssp_weighted(
				graph_heads, 
				graph_tails, 
				graph_weights,
				tile_offsets,
				tile_sizes,
				graph_vertices,
				graph_edges,
				graph_weights_csr,
				graph_degrees,
				source);
		} else {
			sssp(
				graph_heads, 
				graph_tails, 
				tile_offsets,
				//h_graph_mask, 
				//h_updating_graph_mask,
				//is_active_side,
				//is_updating_active_side,
				tile_sizes,
				//distances,
				source);
		}
		}
		bot_best_perform.print_average(NUM_THREADS);
	}

	// Free memory
	free(graph_heads);
	free(graph_tails);
	if (nullptr != graph_weights) {
		free(graph_weights);
		free(graph_vertices);
		free(graph_edges);
		free(graph_weights_csr);
		free(graph_degrees);
	}
	free(tile_offsets);
	free(tile_sizes);
	//free(distances);
	//free(h_graph_mask);
	//free(h_updating_graph_mask);
	//free(is_active_side);
	//free(is_updating_active_side);

	return 0;
}
