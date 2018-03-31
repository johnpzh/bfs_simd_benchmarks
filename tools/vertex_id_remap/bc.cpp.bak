/*
   Re-map vertex ID to new ID. New ID's order is determined according to access order. 
   Those vertices accessed earlier will have a smaller ID.
*/
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
unsigned TILE_WIDTH;
unsigned SIDE_LENGTH;
unsigned NUM_TILES;
unsigned ROW_STEP;
unsigned CHUNK_SIZE_DENSE;
unsigned CHUNK_SIZE_SPARSE;
unsigned CHUNK_SIZE_BLOCK;
unsigned SIZE_BUFFER_MAX;
unsigned T_RATIO;
unsigned WORK_LOAD;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


void print_m512i(__m512i v)
{
	int *a = (int *) &v;
	printf("__m512i:");
	for (int i = 0; i < 16; ++i) {
		printf(" %d", a[i]);
	}
	putchar('\n');
}

inline unsigned get_chunk_size(unsigned amount)
{
	unsigned r = amount / NUM_THREADS / WORK_LOAD;
	if (r) {
		return r;
	} else {
		return 1;
	}
}

void input_weighted(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
		unsigned *&graph_vertices,
		unsigned *&graph_edges,
		unsigned *&graph_degrees,
		unsigned *&graph_weights,
		unsigned *&tile_offsets,
		unsigned *&tile_sizes)
{
	//printf("data: %s\n", filename);
	string prefix = string(filename) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
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

	// Read tile Offsets, and get tile sizes
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
	graph_vertices = (unsigned *) malloc(NNODES * sizeof(unsigned));
	graph_edges = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) malloc(NNODES * sizeof(unsigned));
	graph_weights = (unsigned *) malloc(NEDGES * sizeof(unsigned));

	// For heads and tails
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
		unsigned wt;
		fscanf(fin, "%u %u %u", &n1, &n2, &wt);
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_tails[index] = n2;
		graph_weights[index] = wt;
	}
	fclose(fin);
}

	//For graph CSR
	prefix = string(filename) + "_untiled";

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

void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
		unsigned *&graph_vertices,
		unsigned *&graph_edges,
		unsigned *&graph_degrees,
		unsigned *&tile_offsets,
		unsigned *&tile_sizes)
{
	//printf("data: %s\n", filename);
	string prefix = string(filename) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
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

	// Read tile Offsets, and get tile sizes
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
	graph_vertices = (unsigned *) malloc(NNODES * sizeof(unsigned));
	graph_edges = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) malloc(NNODES * sizeof(unsigned));

	// For heads and tails
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
	}
	fclose(fin);
}

	//For graph CSR
	prefix = string(filename) + "_untiled";

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
///////////////////////////////////////////////////////
// Sparse
inline unsigned update_visited_sparse(
						int *h_graph_visited,
						unsigned *h_graph_queue,
						unsigned queue_size,
						unsigned *graph_degrees)
{
//	unsigned out_degree = 0;
//#pragma omp parallel for reduction(+: out_degree)
//	for (unsigned i = 0; i < queue_size; ++i)
//	{
//		unsigned vertex_id = h_graph_queue[i];
//		out_degree += graph_degrees[vertex_id];
//		h_graph_visited[vertex_id] = 1;
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
		_mm512_i32scatter_epi32(h_graph_visited, vertex_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
	if (remainder) {
		__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512i vertex_id_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, h_graph_queue + bound_i);
		__m512i degrees_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, vertex_id_v, graph_degrees, sizeof(unsigned));
		unsigned sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		out_degree += sum_degrees;
		_mm512_mask_i32scatter_epi32(h_graph_visited, in_range_m, vertex_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
	return out_degree;
}

inline void update_visited_sparse_reverse(
		unsigned *h_graph_queue,
		const unsigned &queue_size,
		int *h_graph_visited,
		float *dependencies,
		float *inverse_num_paths)
{
//#pragma omp parallel for
//	for (unsigned i = 0; i < queue_size; ++i) {
//		unsigned tail_id = h_graph_queue[i];
//		h_graph_visited[tail_id] = 1;
//		dependencies[tail_id] += inverse_num_paths[tail_id];
//	}
	unsigned remainder = queue_size % NUM_P_INT;
	unsigned bound_i = queue_size - remainder;
#pragma omp parallel for
	for (unsigned i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i vertex_id_v = _mm512_load_epi32(h_graph_queue + i);
		_mm512_i32scatter_epi32(h_graph_visited, vertex_id_v, _mm512_set1_epi32(1), sizeof(int));
		__m512 inv_num_paths_v = _mm512_i32gather_ps(vertex_id_v, inverse_num_paths, sizeof(float));
		__m512 dep_v = _mm512_i32gather_ps(vertex_id_v, dependencies, sizeof(float));
		dep_v = _mm512_add_ps(dep_v, inv_num_paths_v);
		_mm512_i32scatter_ps(dependencies, vertex_id_v, dep_v, sizeof(float));
	}
	if (remainder) {
		__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512i vertex_id_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, h_graph_queue + bound_i);
		_mm512_mask_i32scatter_epi32(h_graph_visited, in_range_m, vertex_id_v, _mm512_set1_epi32(1), sizeof(int));
		__m512 inv_num_paths_v = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), in_range_m, vertex_id_v, inverse_num_paths, sizeof(float));
		__m512 dep_v = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), in_range_m, vertex_id_v, dependencies, sizeof(float));
		dep_v = _mm512_mask_add_ps(_mm512_undefined_ps(), in_range_m, dep_v, inv_num_paths_v);
		_mm512_mask_i32scatter_ps(dependencies, in_range_m, vertex_id_v, dep_v, sizeof(float));
	}
}
inline unsigned *to_sparse(
		const unsigned &frontier_size,
		unsigned *h_graph_mask)
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
//#pragma omp parallel for schedule(dynamic, CHUNK_SIZE_BLOCK)
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
//#pragma omp parallel for schedule(dynamic, CHUNK_SIZE_BLOCK)
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
inline unsigned *BFS_kernel_sparse(
				unsigned *graph_vertices,
				unsigned *graph_edges,
				unsigned *graph_degrees,
				int *h_graph_visited,
				unsigned *h_graph_queue,
				unsigned &queue_size,
				unsigned *num_paths)
{
	// From h_graph_queue, get the degrees (para_for)
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
		__mmask16 in_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512i v_ids = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_m, h_graph_queue + bound_i);
		__m512i degrees_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_m, v_ids, graph_degrees, sizeof(unsigned));
		_mm512_mask_store_epi32(degrees + bound_i, in_m, degrees_v);
		unsigned sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		new_queue_size += sum_degrees;
	}
	//for (unsigned i = 0; i < queue_size; ++i) {
	//	degrees[i] = graph_degrees[h_graph_queue[i]];
	//	new_queue_size += degrees[i];
	//}
	if (0 == new_queue_size) {
		_mm_free(degrees);
		queue_size = 0;
		//h_graph_queue = nullptr;
		return nullptr;
	}

	// From degrees, get the offset (stored in degrees) (block_para_for)
	// TODO: blocked parallel for
	unsigned offset_sum = 0;
	for (unsigned i = 0; i < queue_size; ++i) {
		unsigned tmp = degrees[i];
		degrees[i] = offset_sum;
		offset_sum += tmp;
	}

	// From offset, get active vertices (para_for)
	//unsigned *new_frontier_tmp = (unsigned *) malloc(sizeof(unsigned) * new_queue_size);
	unsigned *new_frontier_tmp = (unsigned *) _mm_malloc(sizeof(unsigned) * new_queue_size, ALIGNED_BYTES);
	CHUNK_SIZE_SPARSE = get_chunk_size(queue_size);
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE_SPARSE)
	for (unsigned i = 0; i < queue_size; ++i) {
		unsigned start = h_graph_queue[i];
		unsigned offset = degrees[i];
		unsigned out_degree = graph_degrees[start];
		unsigned base = graph_vertices[start];
		for (unsigned k = 0; k < out_degree; ++k) {
			unsigned end = graph_edges[base + k];
			if (0 == h_graph_visited[end]) {
				//bool unvisited = __sync_bool_compare_and_swap(h_graph_visited + end, 0, 1); //update h_graph_visited
				//if (unvisited) {
				//	new_frontier_tmp[offset + k] = end;
				//} else {
				//	new_frontier_tmp[offset + k] = (unsigned) -1;
				//}
				// Update num_paths
				volatile unsigned old_val;
				volatile unsigned new_val;
				do {
					old_val = num_paths[end];
					new_val = old_val + num_paths[start];
				} while (!__sync_bool_compare_and_swap(num_paths + end, old_val, new_val));
				if (old_val == 0.0) {
					new_frontier_tmp[offset + k] = end;
				} else {
					new_frontier_tmp[offset + k] = (unsigned) -1;
				}
			} else {
				new_frontier_tmp[offset + k] = (unsigned) -1;
			}
		}
	}


	// Refine active vertices, removing visited and redundant (block_para_for)
	unsigned block_size = 1024 * 2;
	unsigned num_blocks = (new_queue_size - 1)/block_size + 1;

	unsigned *nums_in_blocks = NULL;
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
		//free(new_frontier_tmp);
		if (nums_in_blocks) {
			free(nums_in_blocks);
		}
		queue_size = 0;
		return nullptr;
	}

	// Get the final new frontier
	//unsigned *new_frontier = (unsigned *) malloc(sizeof(unsigned) * new_queue_size);
	unsigned *new_frontier = (unsigned *) _mm_malloc(sizeof(unsigned) * new_queue_size, ALIGNED_BYTES);
	if (num_blocks > 1) {
		//TODO: blocked parallel for
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
				__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
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
			__mmask16 in_range_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
			__m512i tmp = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, new_frontier_tmp + bound_i);
			_mm512_mask_store_epi32(new_frontier + bound_i, in_range_m, tmp);
		}
		//unsigned base = 0;
		//for (unsigned i = 0; i < new_queue_size; ++i) {
		//	new_frontier[i] = new_frontier_tmp[base++];
		//}
	}

	// Return the results
	_mm_free(degrees);
	_mm_free(new_frontier_tmp);
	//free(new_frontier_tmp);
	if (nums_in_blocks) {
		free(nums_in_blocks);
	}
	queue_size = new_queue_size;
	return new_frontier;
}
inline unsigned *BFS_sparse(
				unsigned *h_graph_queue,
				unsigned &queue_size,
				unsigned *graph_vertices,
				unsigned *graph_edges,
				unsigned *graph_degrees,
				int *h_graph_visited,
				unsigned *num_paths)
{
	return BFS_kernel_sparse(
				graph_vertices,
				graph_edges,
				graph_degrees,
				h_graph_visited,
				h_graph_queue,
				queue_size,
				num_paths);
}

inline void BFS_kernel_sparse_reverse(
				unsigned *graph_vertices,
				unsigned *graph_edges,
				unsigned *graph_degrees,
				int *h_graph_visited,
				unsigned *h_graph_queue,
				unsigned &queue_size,
				float *dependencies)
{
	// From offset, get active vertices (para_for)
	CHUNK_SIZE_SPARSE = get_chunk_size(queue_size);
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE_SPARSE)
	for (unsigned i = 0; i < queue_size; ++i) {
		unsigned start = h_graph_queue[i];
		unsigned base_edge_i = graph_vertices[start];
		unsigned bound_edge_i = base_edge_i + graph_degrees[start];
		for (unsigned edge_i = base_edge_i; edge_i < bound_edge_i; ++edge_i) {
			unsigned end = graph_edges[edge_i];
			if (0 == h_graph_visited[end]) {
				volatile float old_val;
				volatile float new_val;
				do {
					old_val = dependencies[end];
					new_val = old_val + dependencies[start];
				} while (!__sync_bool_compare_and_swap(
												(int *) (dependencies + end), 
												*((int *) &old_val), 
												*((int *) &new_val)));
			}
		}
	}
}
inline void BFS_sparse_reverse(
				unsigned *h_graph_queue,
				unsigned &queue_size,
				unsigned *graph_vertices,
				unsigned *graph_edges,
				unsigned *graph_degrees,
				int *h_graph_visited,
				float *dependencies)
{
	BFS_kernel_sparse_reverse(
				graph_vertices,
				graph_edges,
				graph_degrees,
				h_graph_visited,
				h_graph_queue,
				queue_size,
				dependencies);
}
// End Sparse
//////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////
// Dense

inline void update_visited_dense(
					unsigned &_frontier_size,
					unsigned &_out_degree,
					int *h_graph_visited,
					unsigned *h_graph_mask,
					int *is_active_side,
					int *is_updating_active_side,
					unsigned *graph_degrees)
{
	unsigned frontier_size = 0;
	unsigned out_degree = 0;
#pragma omp parallel for reduction(+: frontier_size, out_degree)
//#pragma omp parallel for schedule(dynamic, 2) reduction(+: frontier_size, out_degree)
	for (unsigned side_id = 0; side_id < SIDE_LENGTH; ++side_id) {
		if (!is_updating_active_side[side_id]) {
			is_active_side[side_id] = 0;
			continue;
		}
		is_updating_active_side[side_id] = 0;
		is_active_side[side_id] = 1;
		unsigned start_vertex_id = side_id * TILE_WIDTH;
		unsigned bound_vertex_id;
		if (SIDE_LENGTH - 1 != side_id) {
			bound_vertex_id = side_id * TILE_WIDTH + TILE_WIDTH;
		} else {
			bound_vertex_id = NNODES;
		}
		unsigned remainder = (bound_vertex_id - start_vertex_id) % NUM_P_INT;
		bound_vertex_id -= remainder;
		unsigned vertex_id;
		for (vertex_id = start_vertex_id; 
				vertex_id < bound_vertex_id; 
				vertex_id += NUM_P_INT) {
			__m512i updating_flag_v = _mm512_loadu_si512(h_graph_mask + vertex_id);
			__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
			if (!is_updating_m) {
				continue;
			}
			__m512i num_active_v = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), is_updating_m, 1);
			unsigned num_active = _mm512_reduce_add_epi32(num_active_v);
			frontier_size += num_active;
			__m512i out_degrees_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), is_updating_m, graph_degrees + vertex_id);
			out_degree += _mm512_reduce_add_epi32(out_degrees_v);
			_mm512_mask_storeu_epi32(h_graph_visited + vertex_id, is_updating_m, _mm512_set1_epi32(1));
		}

		if (0 == remainder) {
			continue;
		}
		unsigned short in_range_m_t = (unsigned short) 0xFFFF >> (NUM_P_INT - remainder);
		__mmask16 in_range_m = (__mmask16) in_range_m_t;
		__m512i updating_flag_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), in_range_m, h_graph_mask + vertex_id);
		__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
		if (!is_updating_m) {
			continue;
		}
		__m512i num_active_v = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), is_updating_m, 1);
		unsigned num_active = _mm512_reduce_add_epi32(num_active_v);
		frontier_size += num_active;
		__m512i out_degrees_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), is_updating_m, graph_degrees + vertex_id);
		out_degree += _mm512_reduce_add_epi32(out_degrees_v);
		_mm512_mask_storeu_epi32(h_graph_visited + vertex_id, is_updating_m, _mm512_set1_epi32(1));
		//for (unsigned vertex_id = side_id * TILE_WIDTH; vertex_id < bound_vertex_id; ++vertex_id) {
		//	if (1 == h_graph_mask[vertex_id]) {
		//		frontier_size++;
		//		out_degree += graph_degrees[vertex_id];
		//		h_graph_visited[vertex_id] = 1;
		//	}
		//}
	}

	_frontier_size = frontier_size;
	_out_degree = out_degree;
}

void update_visited_dense_reverse(
		unsigned *h_graph_mask,
		int *h_graph_visited,
		float *dependencies,
		float *inverse_num_paths)
{
//#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
//	for (unsigned i = 0; i < NNODES; ++i) {
//		if (h_graph_mask[i]) {
//			h_graph_visited[i] = 1;
//			dependencies[i] += inverse_num_paths[i];
//		}
//	}
	unsigned remainder = NNODES % NUM_P_INT;
	unsigned bound_i = NNODES - remainder;
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE_DENSE)
	for (unsigned i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i mask_v = _mm512_load_epi32(h_graph_mask + i);
		__mmask16 is_set_v = _mm512_test_epi32_mask(mask_v, _mm512_set1_epi32(-1));
		if (!is_set_v) {
			continue;
		}
		_mm512_mask_store_epi32(h_graph_visited + i, is_set_v, _mm512_set1_epi32(1));
		__m512 inv_num_paths_v = _mm512_mask_load_ps(_mm512_undefined_ps(), is_set_v, inverse_num_paths + i);
		__m512 dep_v = _mm512_mask_load_ps(_mm512_undefined_ps(), is_set_v, dependencies + i);
		dep_v = _mm512_mask_add_ps(dep_v, is_set_v, dep_v, inv_num_paths_v);
		_mm512_mask_store_ps(dependencies + i, is_set_v, dep_v);
	}
	if (remainder) {
		__mmask16 in_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512i mask_v = _mm512_mask_load_epi32(_mm512_set1_epi32(0), in_m, h_graph_mask + bound_i);
		__mmask16 is_set_v = _mm512_test_epi32_mask(mask_v, _mm512_set1_epi32(-1));
		if (!is_set_v) {
			return;
		}
		_mm512_mask_store_epi32(h_graph_visited + bound_i, is_set_v, _mm512_set1_epi32(1));
		__m512 inv_num_paths_v = _mm512_mask_load_ps(_mm512_undefined_ps(), is_set_v, inverse_num_paths + bound_i);
		__m512 dep_v = _mm512_mask_load_ps(_mm512_undefined_ps(), is_set_v, dependencies + bound_i);
		dep_v = _mm512_mask_add_ps(dep_v, is_set_v, dep_v, inv_num_paths_v);
		_mm512_mask_store_ps(dependencies + bound_i, is_set_v, dep_v);
	}
}

unsigned *to_dense(
		int *is_active_side, 
		unsigned *h_graph_queue, 
		unsigned frontier_size)
{
	unsigned *new_mask = (unsigned *) calloc(NNODES, sizeof(unsigned));
	unsigned remainder = frontier_size % NUM_P_INT;
	unsigned bound_i = frontier_size - remainder;
#pragma omp parallel for
	for (unsigned i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i v_ids_v = _mm512_load_epi32(h_graph_queue + i);
		_mm512_i32scatter_epi32(new_mask, v_ids_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i tw_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(v_ids_v, tw_v);
		_mm512_i32scatter_epi32(is_active_side, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
	if (remainder) {
		__mmask16 in_m = (__mmask16) ((unsigned short) 0xFFFF >> (NUM_P_INT - remainder));
		__m512i v_ids_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_m, h_graph_queue + bound_i);
		_mm512_mask_i32scatter_epi32(new_mask, in_m, v_ids_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i tw_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_mask_div_epi32(_mm512_undefined_epi32(), in_m, v_ids_v, tw_v);
		_mm512_mask_i32scatter_epi32(is_active_side, in_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
//	for (unsigned i = 0; i < frontier_size; ++i) {
//		unsigned vertex_id = h_graph_queue[i];
//		new_mask[vertex_id] = 1;
//		is_active_side[vertex_id / TILE_WIDTH] = 1;
//	}
	return new_mask;
}
//inline void bfs_kernel_dense(
//		const unsigned &start_edge_i,
//		const unsigned &bound_edge_i,
//		unsigned *h_graph_heads,
//		unsigned *h_graph_tails,
//		unsigned *h_graph_mask,
//		unsigned *h_updating_graph_mask,
//		int *h_graph_visited,
//		//unsigned *h_graph_parents,
//		//int *h_cost,
//		int *is_updating_active_side,
//		unsigned *num_paths)
//{
//	for (unsigned edge_i = start_edge_i; edge_i < bound_edge_i; ++edge_i) {
//		unsigned head = h_graph_heads[edge_i];
//		if (0 == h_graph_mask[head]) {
//			//++edge_i;
//			continue;
//		}
//		unsigned end = h_graph_tails[edge_i];
//		//if ((unsigned) -1 == h_graph_parents[end]) {
//		//	h_cost[end] = h_cost[head] + 1;
//		//	h_updating_graph_mask[end] = 1;
//		//	is_updating_active_side[end/TILE_WIDTH] = 1;
//		//	h_graph_parents[end] = head; // addition
//		//}
//		if (0 == h_graph_visited[end]) {
//			volatile unsigned old_val;
//			volatile unsigned new_val;
//			do {
//				old_val = num_paths[end];
//				new_val = old_val + num_paths[head];
//			} while (!__sync_bool_compare_and_swap(num_paths + end, old_val, new_val));
//			if (old_val == 0.0) {
//				h_updating_graph_mask[end] = 1;
//				is_updating_active_side[end/TILE_WIDTH] = 1;
//			}
//		}
//	}
//}

// Scan the data, accumulate the values with the same index.
// Then, store the cumulative sum to the last element in the data with the same index.
void scan_for_gather_add_scatter_conflict_safe_epi32(
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

inline void bfs_kernel_dense(
		unsigned *heads_buffer,
		unsigned *tails_buffer,
		const unsigned &size_buffer,
		unsigned *h_graph_mask,
		unsigned *h_updating_graph_mask,
		int *h_graph_visited,
		//unsigned *h_graph_parents,
		//int *h_cost,
		int *is_updating_active_side,
		unsigned *num_paths)
{
	unsigned remainder = size_buffer % NUM_P_INT;
	unsigned bound_edge_i = size_buffer - remainder;
	unsigned edge_i;
	for (edge_i = 0; edge_i < bound_edge_i; edge_i += NUM_P_INT) {
		__m512i head_v = _mm512_load_epi32(heads_buffer + edge_i);
		__m512i active_flag_v = _mm512_i32gather_epi32(head_v, h_graph_mask, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			continue;
		}
		__m512i tail_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, tails_buffer + edge_i);
		__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_visited, sizeof(int));
		__mmask16 not_visited_m = _mm512_testn_epi32_mask(visited_flag_v, _mm512_set1_epi32(1));
		//__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_parents, sizeof(int));
		//__mmask16 not_visited_m = _mm512_cmpeq_epi32_mask(visited_flag_v, _mm512_set1_epi32(-1));
		if (!not_visited_m) {
			continue;
		}
		__m512i num_paths_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, num_paths, sizeof(int));
		scan_for_gather_add_scatter_conflict_safe_epi32(num_paths_head_v, tail_v);
		__m512i num_paths_tail_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, tail_v, num_paths, sizeof(int));
		num_paths_tail_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, num_paths_tail_v, num_paths_head_v);
		_mm512_mask_i32scatter_epi32(num_paths, not_visited_m, tail_v, num_paths_tail_v, sizeof(int));
		//__m512i cost_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, h_cost, sizeof(int));
		//__m512i cost_tail_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, cost_head_v, _mm512_set1_epi32(1));
		//_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, tail_v, cost_tail_v, sizeof(int));
		_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(tail_v, TILE_WIDTH_v);
		_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
		//_mm512_mask_i32scatter_epi32(h_graph_parents, not_visited_m, tail_v, head_v, sizeof(unsigned));
		//_mm512_mask_i32scatter_epi32(h_graph_visited, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(unsigned));
	}

	if (0 == remainder) {
		return;
	}
	unsigned short in_range_m_t = (unsigned short) 0xFFFF >> (NUM_P_INT - remainder);
	__mmask16 in_range_m = (__mmask16) in_range_m_t;
	__m512i head_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, heads_buffer + edge_i);
	__m512i active_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, head_v, h_graph_mask, sizeof(int));
	__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
	if (!is_active_m) {
		return;
	}
	__m512i tail_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, tails_buffer + edge_i);
	__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_visited, sizeof(int));
	__mmask16 not_visited_m = _mm512_testn_epi32_mask(visited_flag_v, _mm512_set1_epi32(1));
	//__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_parents, sizeof(int));
	//__mmask16 not_visited_m = _mm512_cmpeq_epi32_mask(visited_flag_v, _mm512_set1_epi32(-1));
	if (!not_visited_m) {
		return;
	}
	__m512i num_paths_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, num_paths, sizeof(int));
	scan_for_gather_add_scatter_conflict_safe_epi32(num_paths_head_v, tail_v);
	__m512i num_paths_tail_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, tail_v, num_paths, sizeof(int));
	num_paths_tail_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, num_paths_tail_v, num_paths_head_v);
	_mm512_mask_i32scatter_epi32(num_paths, not_visited_m, tail_v, num_paths_tail_v, sizeof(int));
	//__m512i cost_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, h_cost, sizeof(int));
	//__m512i cost_tail_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, cost_head_v, _mm512_set1_epi32(1));
	//_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, tail_v, cost_tail_v, sizeof(int));
	_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(int));
	__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
	__m512i side_id_v = _mm512_div_epi32(tail_v, TILE_WIDTH_v);
	_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	//_mm512_mask_i32scatter_epi32(h_graph_parents, not_visited_m, tail_v, head_v, sizeof(unsigned));
	//_mm512_mask_i32scatter_epi32(h_graph_visited, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(unsigned));
}
inline void scheduler_dense(
		const unsigned &start_row_index,
		const unsigned &tile_step,
		unsigned *h_graph_heads,
		unsigned *h_graph_tails,
		unsigned *heads_buffer,
		unsigned *tails_buffer,
		unsigned *h_graph_mask,
		unsigned *h_updating_graph_mask,
		int *h_graph_visited,
		//unsigned *h_graph_parents,
		//int *h_cost,
		unsigned *tile_offsets,
		//int *is_empty_tile,
		unsigned *tile_sizes,
		int *is_active_side,
		int *is_updating_active_side,
		unsigned *num_paths)
{
	unsigned start_tile_id = start_row_index * SIDE_LENGTH;
	unsigned end_tile_id = start_tile_id + tile_step * SIDE_LENGTH;
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
		unsigned bound_tile_id = tile_index + tile_step;
		unsigned tid = omp_get_thread_num();
		unsigned *heads_buffer_base = heads_buffer + tid * SIZE_BUFFER_MAX;
		unsigned *tails_buffer_base = tails_buffer + tid * SIZE_BUFFER_MAX;
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
						memcpy(heads_buffer_base + size_buffer, h_graph_heads + edge_i, remain * sizeof(unsigned));
						memcpy(tails_buffer_base + size_buffer, h_graph_tails + edge_i, remain * sizeof(unsigned));
						edge_i += remain;
						capacity -= remain;
						size_buffer += remain;
						remain = 0;
					} else {
						// Fill the buffer to full
						memcpy(heads_buffer_base + size_buffer, h_graph_heads + edge_i, capacity * sizeof(unsigned));
						memcpy(tails_buffer_base + size_buffer, h_graph_tails + edge_i, capacity * sizeof(unsigned));
						edge_i += capacity;
						remain -= capacity;
						size_buffer += capacity;
						capacity = 0;
					}
				} else {
					// Process the full buffer
					bfs_kernel_dense(
							heads_buffer_base,
							tails_buffer_base,
							size_buffer,
							h_graph_mask,
							h_updating_graph_mask,
							h_graph_visited,
							//h_graph_parents,
							//h_cost,
							is_updating_active_side,
							num_paths);
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}
			
		}
		// Process the remains in buffer
		bfs_kernel_dense(
				heads_buffer_base,
				tails_buffer_base,
				size_buffer,
				h_graph_mask,
				h_updating_graph_mask,
				h_graph_visited,
				//h_graph_parents,
				//h_cost,
				is_updating_active_side,
				num_paths);
	}
	//for (unsigned tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
	//	unsigned bound_tile_id = tile_index + tile_step;
	//	for (unsigned tile_id = tile_index; tile_id < bound_tile_id; ++tile_id) {
	//		unsigned row_id = (tile_id - start_tile_id) % tile_step + start_row_index;
	//		//if (is_empty_tile[tile_id] || !is_active_side[row_id]) {
	//		//	continue;
	//		//}
	//		if (!tile_sizes[tile_id] || !is_active_side[row_id]) {
	//			continue;
	//		}
	//		// Kernel
	//		unsigned bound_edge_i;
	//		if (NUM_TILES - 1 != tile_id) {
	//			bound_edge_i = tile_offsets[tile_id + 1];
	//		} else {
	//			bound_edge_i = NEDGES;
	//		}
	//		bfs_kernel_dense(
	//				tile_offsets[tile_id],
	//				bound_edge_i,
	//				h_graph_heads,
	//				h_graph_tails,
	//				h_graph_mask,
	//				h_updating_graph_mask,
	//				h_graph_visited,
	//				//h_graph_parents,
	//				//h_cost,
	//				is_updating_active_side,
	//				num_paths);
	//	}

	//}
}

inline unsigned *BFS_dense(
		unsigned *h_graph_heads,
		unsigned *h_graph_tails,
		unsigned *h_graph_mask,
		//int *h_updating_graph_mask,
		int *h_graph_visited,
		//unsigned *h_graph_parents,
		//int *h_cost,
		unsigned *tile_offsets,
		//int *is_empty_tile,
		unsigned *tile_sizes,
		int *is_active_side,
		int *is_updating_active_side,
		unsigned *num_paths)
{
	unsigned *heads_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned *tails_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);

	unsigned remainder = SIDE_LENGTH % ROW_STEP;
	unsigned bound_side_id = SIDE_LENGTH - remainder;
	//unsigned *new_mask = (unsigned *) calloc(NNODES, sizeof(unsigned));
	unsigned *new_mask = (unsigned *) _mm_malloc(NNODES * sizeof(unsigned), ALIGNED_BYTES);
	memset(new_mask, 0, NNODES * sizeof(unsigned));
	unsigned side_id;
	for (side_id = 0; side_id < bound_side_id; side_id += ROW_STEP) {
		scheduler_dense(
				//side_id * SIDE_LENGTH,\
				//(side_id + ROW_STEP) * SIDE_LENGTH,
				side_id,
				ROW_STEP,
				h_graph_heads,
				h_graph_tails,
				heads_buffer,
				tails_buffer,
				h_graph_mask,
				new_mask,
				h_graph_visited,
				//h_graph_parents,
				//h_cost,
				tile_offsets,
				//is_empty_tile,
				tile_sizes,
				is_active_side,
				is_updating_active_side,
				num_paths);
	}
	scheduler_dense(
			//side_id * SIDE_LENGTH,\
			//NUM_TILES,
			side_id,
			remainder,
			h_graph_heads,
			h_graph_tails,
			heads_buffer,
			tails_buffer,
			h_graph_mask,
			new_mask,
			h_graph_visited,
			//h_graph_parents,
			//h_cost,
			tile_offsets,
			//is_empty_tile,
			tile_sizes,
			is_active_side,
			is_updating_active_side,
			num_paths);
	_mm_free(heads_buffer);
	_mm_free(tails_buffer);
	return new_mask;
}

//inline void bfs_kernel_dense_reverse(
//		const unsigned &start_edge_i,
//		const unsigned &bound_edge_i,
//		unsigned *h_graph_heads,
//		unsigned *h_graph_tails,
//		unsigned *h_graph_mask,
//		//unsigned *h_updating_graph_mask,
//		int *h_graph_visited,
//		//unsigned *h_graph_parents,
//		//int *h_cost,
//		//int *is_updating_active_side,
//		float *dependencies)
//{
//	for (unsigned edge_i = start_edge_i; edge_i < bound_edge_i; ++edge_i) {
//		unsigned head = h_graph_heads[edge_i];
//		if (0 == h_graph_mask[head]) {
//			//++edge_i;
//			continue;
//		}
//		unsigned end = h_graph_tails[edge_i];
//		if (0 == h_graph_visited[end]) {
//			volatile float old_val;
//			volatile float new_val;
//			do {
//				old_val = dependencies[end];
//				new_val = old_val + dependencies[head];
//			} while (!__sync_bool_compare_and_swap(
//											(int *) (dependencies + end), 
//											*((int *) &old_val), 
//											*((int *) &new_val)));
//			//do {
//			//	old_val = num_paths[end];
//			//	new_val = old_val + num_paths[head];
//			//} while (!__sync_bool_compare_and_swap(num_paths + end, old_val, new_val));
//		}
//	}
//}

// Scan the data, accumulate the values with the same index.
// Then, store the cumulative sum to the last element in the data with the same index.
void scan_for_gather_add_scatter_conflict_safe_ps(
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

inline void bfs_kernel_dense_reverse(
		unsigned *heads_buffer,
		unsigned *tails_buffer,
		const unsigned &size_buffer,
		unsigned *h_graph_mask,
		//int *h_updating_graph_mask,
		int *h_graph_visited,
		//unsigned *h_graph_parents,
		//int *h_cost,
		//int *is_updating_active_side,
		float *dependencies)
{
	unsigned remainder = size_buffer % NUM_P_INT;
	unsigned bound_edge_i = size_buffer - remainder;
	unsigned edge_i;
	for (edge_i = 0; edge_i < bound_edge_i; edge_i += NUM_P_INT) {
		__m512i head_v = _mm512_load_epi32(heads_buffer + edge_i);
		__m512i active_flag_v = _mm512_i32gather_epi32(head_v, h_graph_mask, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			continue;
		}
		__m512i tail_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, tails_buffer + edge_i);
		__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_visited, sizeof(int));
		__mmask16 not_visited_m = _mm512_testn_epi32_mask(visited_flag_v, _mm512_set1_epi32(1));
		//__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_parents, sizeof(int));
		//__mmask16 not_visited_m = _mm512_cmpeq_epi32_mask(visited_flag_v, _mm512_set1_epi32(-1));
		if (!not_visited_m) {
			continue;
		}
		__m512 dependencies_head_v = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), not_visited_m, head_v, dependencies, sizeof(float));
		scan_for_gather_add_scatter_conflict_safe_ps(dependencies_head_v, tail_v);
		__m512 dependencies_tail_v = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), not_visited_m, tail_v, dependencies, sizeof(float));
		dependencies_tail_v = _mm512_mask_add_ps(_mm512_undefined_ps(), not_visited_m, dependencies_tail_v, dependencies_head_v);
		_mm512_mask_i32scatter_ps(dependencies, not_visited_m, tail_v, dependencies_tail_v, sizeof(float));
		//__m512i cost_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, h_cost, sizeof(int));
		//__m512i cost_tail_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, cost_head_v, _mm512_set1_epi32(1));
		//_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, tail_v, cost_tail_v, sizeof(int));
		//_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(int));
		//__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
		//__m512i side_id_v = _mm512_div_epi32(tail_v, TILE_WIDTH_v);
		//_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
		//_mm512_mask_i32scatter_epi32(h_graph_parents, not_visited_m, tail_v, head_v, sizeof(unsigned));
		//_mm512_mask_i32scatter_epi32(h_graph_visited, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(unsigned));
	}

	if (0 == remainder) {
		return;
	}
	unsigned short in_range_m_t = (unsigned short) 0xFFFF >> (NUM_P_INT - remainder);
	__mmask16 in_range_m = (__mmask16) in_range_m_t;
	__m512i head_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, heads_buffer + edge_i);
	__m512i active_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, head_v, h_graph_mask, sizeof(int));
	__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
	if (!is_active_m) {
		return;
	}
	__m512i tail_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, tails_buffer + edge_i);
	__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_visited, sizeof(int));
	__mmask16 not_visited_m = _mm512_testn_epi32_mask(visited_flag_v, _mm512_set1_epi32(1));
	//__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_parents, sizeof(int));
	//__mmask16 not_visited_m = _mm512_cmpeq_epi32_mask(visited_flag_v, _mm512_set1_epi32(-1));
	if (!not_visited_m) {
		return;
	}
	__m512 dependencies_head_v = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), not_visited_m, head_v, dependencies, sizeof(float));
	scan_for_gather_add_scatter_conflict_safe_ps(dependencies_head_v, tail_v);
	__m512 dependencies_tail_v = _mm512_mask_i32gather_ps(_mm512_undefined_ps(), not_visited_m, tail_v, dependencies, sizeof(float));
	dependencies_tail_v = _mm512_mask_add_ps(_mm512_undefined_ps(), not_visited_m, dependencies_tail_v, dependencies_head_v);
	_mm512_mask_i32scatter_ps(dependencies, not_visited_m, tail_v, dependencies_tail_v, sizeof(float));
	//__m512i cost_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, h_cost, sizeof(int));
	//__m512i cost_tail_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, cost_head_v, _mm512_set1_epi32(1));
	//_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, tail_v, cost_tail_v, sizeof(int));
	//_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(int));
	//__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
	//__m512i side_id_v = _mm512_div_epi32(tail_v, TILE_WIDTH_v);
	//_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	//_mm512_mask_i32scatter_epi32(h_graph_parents, not_visited_m, tail_v, head_v, sizeof(unsigned));
	//_mm512_mask_i32scatter_epi32(h_graph_visited, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(unsigned));
}
inline void scheduler_dense_reverse(
		const unsigned &start_row_index,
		const unsigned &tile_step,
		unsigned *h_graph_heads,
		unsigned *h_graph_tails,
		unsigned *heads_buffer,
		unsigned *tails_buffer,
		unsigned *h_graph_mask,
		//unsigned *h_updating_graph_mask,
		int *h_graph_visited,
		//unsigned *h_graph_parents,
		//int *h_cost,
		unsigned *tile_offsets,
		//int *is_empty_tile,
		unsigned *tile_sizes,
		//int *is_active_side,
		//int *is_updating_active_side,
		float *dependencies)
{
	unsigned start_tile_id = start_row_index * SIDE_LENGTH;
	//unsigned bound_row_id = start_row_index + tile_step;
	unsigned end_tile_id = start_tile_id + tile_step * SIDE_LENGTH;
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
		unsigned bound_tile_id = tile_index + tile_step;
		unsigned tid = omp_get_thread_num();
		unsigned *heads_buffer_base = heads_buffer + tid * SIZE_BUFFER_MAX;
		unsigned *tails_buffer_base = tails_buffer + tid * SIZE_BUFFER_MAX;
		unsigned size_buffer = 0;
		unsigned capacity = SIZE_BUFFER_MAX;
		for (unsigned tile_id = tile_index; tile_id < bound_tile_id; ++tile_id) {
			unsigned row_id = (tile_id - start_tile_id) % tile_step + start_row_index;
			//if (0 == tile_sizes[tile_id] || !is_active_side[row_id]) {
			//	continue;
			//}
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
						memcpy(heads_buffer_base + size_buffer, h_graph_heads + edge_i, remain * sizeof(unsigned));
						memcpy(tails_buffer_base + size_buffer, h_graph_tails + edge_i, remain * sizeof(unsigned));
						edge_i += remain;
						capacity -= remain;
						size_buffer += remain;
						remain = 0;
					} else {
						// Fill the buffer to full
						memcpy(heads_buffer_base + size_buffer, h_graph_heads + edge_i, capacity * sizeof(unsigned));
						memcpy(tails_buffer_base + size_buffer, h_graph_tails + edge_i, capacity * sizeof(unsigned));
						edge_i += capacity;
						remain -= capacity;
						size_buffer += capacity;
						capacity = 0;
					}
				} else {
					// Process the full buffer
					bfs_kernel_dense_reverse(
							heads_buffer_base,
							tails_buffer_base,
							size_buffer,
							h_graph_mask,
							//h_updating_graph_mask,
							h_graph_visited,
							//h_graph_parents,
							//h_cost,
							//is_updating_active_side,
							dependencies);
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}
			
		}
		// Process the remains in buffer
		bfs_kernel_dense_reverse(
				heads_buffer_base,
				tails_buffer_base,
				size_buffer,
				h_graph_mask,
				//h_updating_graph_mask,
				h_graph_visited,
				//h_graph_parents,
				//h_cost,
				//is_updating_active_side,
				dependencies);
	}
//	for (unsigned tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
//		unsigned bound_tile_id = tile_index + tile_step;
//		for (unsigned tile_id = tile_index; tile_id < bound_tile_id; ++tile_id) {
//			unsigned row_id = (tile_id - start_tile_id) % tile_step + start_row_index;
//			//if (!tile_sizes[tile_id] || !is_active_side[row_id]) {
//			//	continue;
//			//}
//			if (!tile_sizes[tile_id]) {
//				continue;
//			}
//			// Kernel
//			unsigned bound_edge_i;
//			if (NUM_TILES - 1 != tile_id) {
//				bound_edge_i = tile_offsets[tile_id + 1];
//			} else {
//				bound_edge_i = NEDGES;
//			}
//			bfs_kernel_dense_reverse(
//					tile_offsets[tile_id],
//					bound_edge_i,
//					h_graph_heads,
//					h_graph_tails,
//					h_graph_mask,
//					//h_updating_graph_mask,
//					h_graph_visited,
//					//h_graph_parents,
//					//h_cost,
//					//is_updating_active_side,
//					dependencies);
//		}
//
//	}
}

inline void BFS_dense_reverse(
		unsigned *h_graph_heads,
		unsigned *h_graph_tails,
		unsigned *h_graph_mask,
		//int *h_updating_graph_mask,
		int *h_graph_visited,
		//unsigned *h_graph_parents,
		//int *h_cost,
		unsigned *tile_offsets,
		//int *is_empty_tile,
		unsigned *tile_sizes,
		//int *is_active_side,
		//int *is_updating_active_side,
		float *dependencies)
{
	unsigned *heads_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned *tails_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);

	unsigned remainder = SIDE_LENGTH % ROW_STEP;
	unsigned bound_side_id = SIDE_LENGTH - remainder;
	//unsigned *new_mask = (unsigned *) calloc(NNODES, sizeof(unsigned));
	unsigned side_id;
	for (side_id = 0; side_id < bound_side_id; side_id += ROW_STEP) {
		scheduler_dense_reverse(
				//side_id * SIDE_LENGTH,\
				//(side_id + ROW_STEP) * SIDE_LENGTH,
				side_id,
				ROW_STEP,
				h_graph_heads,
				h_graph_tails,
				heads_buffer,
				tails_buffer,
				h_graph_mask,
				//new_mask,
				h_graph_visited,
				//h_graph_parents,
				//h_cost,
				tile_offsets,
				//is_empty_tile,
				tile_sizes,
				//is_active_side,
				//is_updating_active_side,
				dependencies);
	}
	scheduler_dense_reverse(
			//side_id * SIDE_LENGTH,\
			//NUM_TILES,
			side_id,
			remainder,
			h_graph_heads,
			h_graph_tails,
			heads_buffer,
			tails_buffer,
			h_graph_mask,
			//new_mask,
			h_graph_visited,
			//h_graph_parents,
			//h_cost,
			tile_offsets,
			//is_empty_tile,
			tile_sizes,
			//is_active_side,
			//is_updating_active_side,
			dependencies);
	_mm_free(heads_buffer);
	_mm_free(tails_buffer);

	//return new_mask;
}
// End Dense
////////////////////////////////////////////////////////////////////////

double dense_time;
double to_dense_time;
double sparse_time;
double to_sparse_time;
double update_time;
double other_time;
double run_time;

void print_time()
{
	auto percent = [=] (double t) {
		return t/run_time * 100.0;
	};
	printf("dense_time: %f (%.2f%%)\n", dense_time, percent(dense_time));
	printf("to_dense_time: %f (%.2f%%)\n", to_dense_time, percent(to_dense_time));
	printf("sparse_time: %f (%.2f%%)\n", sparse_time, percent(sparse_time));
	printf("to_sparse_time: %f (%.2f%%)\n", to_sparse_time, percent(to_sparse_time));
	printf("update_time: %f (%.2f%%)\n", update_time, percent(update_time));
	printf("other_time: %f (%.2f%%)\n", other_time, percent(other_time));
	printf("=========================\n");
}

void add_mask2map(
			unsigned *vertex_map,
			unsigned &top_index,
			unsigned *h_graph_mask)
{
	for (unsigned i = 0; i < NNODES; ++i) {
		if (!h_graph_mask[i]) {
			continue;
		}
		//if ((unsigned) -1 != vertex_map[i]) {
		//	printf("Error: add_mask2map: double-parked, vertex_map[%u]: %u.\n", i, vertex_map[i]);
		//	exit(2);
		//}
		vertex_map[i] = top_index++;
	}
}

void add_remainder2map(
		unsigned *vertex_map,
		unsigned &top_index)
{
	for (unsigned i =0; i < NNODES; ++i) {
		if ((unsigned) -1 == vertex_map[i]) {
			vertex_map[i] = top_index++;
		}
	}
}


void BC(
		unsigned *graph_heads, 
		unsigned *graph_tails, 
		unsigned *graph_vertices,
		unsigned *graph_edges,
		unsigned *graph_degrees,
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		const unsigned &source,
		unsigned *vertex_map)
{
	dense_time = 0.0;
	to_dense_time = 0.0;
	sparse_time = 0.0 ;
	to_sparse_time = 0.0;
	update_time = 0.0;
	unsigned top_index = 0;

	omp_set_num_threads(NUM_THREADS);
	//unsigned *num_paths = (unsigned *) calloc(NNODES, sizeof(unsigned));
	unsigned *num_paths = (unsigned *) _mm_malloc(NNODES * sizeof(unsigned), ALIGNED_BYTES);
	memset(num_paths, 0, NNODES * sizeof(unsigned));
	//int *h_graph_visited = (int *) calloc(NNODES, sizeof(int));
	int *h_graph_visited = (int *) _mm_malloc(NNODES * sizeof(unsigned), ALIGNED_BYTES);
	memset(h_graph_visited, 0, NNODES * sizeof(unsigned));
	int *is_active_side = (int *) calloc(SIDE_LENGTH, sizeof(int));
	int *is_updating_active_side = (int *) calloc(SIDE_LENGTH, sizeof(int));
	unsigned *h_graph_mask = nullptr;
	//float *dependencies = (float *) calloc(NNODES, sizeof(float));
	float *dependencies = (float *) _mm_malloc(NNODES * sizeof(float), ALIGNED_BYTES);
	memset(dependencies, 0, NNODES * sizeof(float));
	vector<unsigned *> frontiers;
	vector<unsigned> frontier_sizes;
	vector<bool> is_dense_frontier;
	unsigned frontier_size;
	unsigned out_degree;

	num_paths[source] = 1;
	// First is the Sparse
	double time_now = omp_get_wtime();
	double last_time = omp_get_wtime();
	frontier_size = 1;
	h_graph_visited[source] = 1;
	unsigned *h_graph_queue = (unsigned *) _mm_malloc(frontier_size * sizeof(unsigned), ALIGNED_BYTES);
	h_graph_queue[0] = source;
	frontiers.push_back(h_graph_queue);
	is_dense_frontier.push_back(false);
	frontier_sizes.push_back(1);

	vertex_map[source] = top_index++;

	double start_time = omp_get_wtime();
	bool last_is_dense = false;
	// First Phase
	// According the sum, determing to run Sparse or Dense, and then change the last_is_dense.
	unsigned bfs_threshold = NEDGES / T_RATIO;
	while (true) {
			if (!last_is_dense) {
				last_time = omp_get_wtime();
				free(is_active_side);
				is_active_side = (int *) calloc(NNODES, sizeof(int));
				h_graph_mask = to_dense(
					is_active_side, 
					h_graph_queue, 
					frontier_size);
				to_dense_time += omp_get_wtime() - last_time;
			}
			last_time = omp_get_wtime();
			unsigned *new_mask = BFS_dense(
					graph_heads,
					graph_tails,
					h_graph_mask,
					//h_updating_graph_mask,
					h_graph_visited,
					//h_graph_parents,
					//h_cost,
					tile_offsets,
					//is_empty_tile,
					tile_sizes,
					is_active_side,
					is_updating_active_side,
					num_paths);
			if (!last_is_dense) {
				free(h_graph_mask);
			}
			h_graph_mask = new_mask;
			last_is_dense = true;
			frontiers.push_back(h_graph_mask);
			is_dense_frontier.push_back(last_is_dense);
			dense_time += omp_get_wtime() - last_time;

			add_mask2map(
					vertex_map,
					top_index,
					h_graph_mask);

		// Update h_graph_visited; Get the sum again.
		if (last_is_dense) {
			last_time = omp_get_wtime();
			update_visited_dense(
					frontier_size,
					out_degree,
					h_graph_visited,
					h_graph_mask,
					is_active_side,
					is_updating_active_side,
					graph_degrees);
			frontier_sizes.push_back(frontier_size);
			update_time += omp_get_wtime() - last_time;
			if (0 == frontier_size) {
				break;
			}
		} else {
			last_time = omp_get_wtime();
			frontier_sizes.push_back(frontier_size);
			if (0 == frontier_size) {
				break;
			}
			out_degree = update_visited_sparse(
										h_graph_visited,
										h_graph_queue,
										frontier_size,
										graph_degrees);
			update_time += omp_get_wtime() - last_time;
		}
	}
	add_remainder2map(
		vertex_map,
		top_index);


	double first_phase_time = omp_get_wtime() - time_now;
	time_now = omp_get_wtime();
		

	// Free memory
	for (auto f = frontiers.begin(); f != frontiers.end(); ++f) {
		_mm_free(*f);
	}
	//for (unsigned i = 0; i < frontiers.size(); ++i) {
	//	printf("frontiers[%u]:\n", i);
	//	for (unsigned v_i = 0; v_i < NNODES; ++v_i) {
	//		if (frontiers[i][v_i] == 1) {
	//			printf(" %u", v_i);
	//		}
	//	}
	//	putchar('\n');
	//	_mm_free(frontiers[i]);
	//}
	frontiers.clear();
	//free(num_paths);
	_mm_free(num_paths);
	//free(h_graph_visited);
	_mm_free(h_graph_visited);
	//free(dependencies);
	_mm_free(dependencies);
	free(is_active_side);
	free(is_updating_active_side);
}

void make_up_data_weighted(
				unsigned *vertex_map,
				unsigned *graph_heads,
				unsigned *graph_tails,
				unsigned *graph_weights,
				char *filename)
{
	puts("Writing...");
	string fname = string(filename) + "_reorder";
	FILE *fout = fopen(fname.c_str(), "w");
	fprintf(fout, "%u %u\n", NNODES, NEDGES);
	for (unsigned e_i = 0; e_i < NEDGES; ++e_i) {
		unsigned head = graph_heads[e_i];
		unsigned tail = graph_tails[e_i];
		unsigned weight = graph_weights[e_i];
		unsigned new_head = vertex_map[head];
		unsigned new_tail = vertex_map[tail];
		++new_head;
		++new_tail;
		fprintf(fout, "%u %u %u\n", new_head, new_tail, weight);
	}
	puts("Done.");
}

void make_up_data(
				unsigned *vertex_map,
				unsigned *graph_heads,
				unsigned *graph_tails,
				char *filename)
{
	puts("Writing...");
	string fname = string(filename) + "_reorder";
	FILE *fout = fopen(fname.c_str(), "w");
	fprintf(fout, "%u %u\n", NNODES, NEDGES);
	for (unsigned e_i = 0; e_i < NEDGES; ++e_i) {
		unsigned head = graph_heads[e_i];
		unsigned tail = graph_tails[e_i];
		unsigned new_head = vertex_map[head];
		unsigned new_tail = vertex_map[tail];
		++new_head;
		++new_tail;
		fprintf(fout, "%u %u\n", new_head, new_tail);
	}
	puts("Done.");
}

int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 3) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
		ROW_STEP = strtoul(argv[3], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec_combine/soc-pokec";
		//filename = "/sciclone/scr-mlt/zpeng01/pokec_combine/soc-pokec";
		TILE_WIDTH = 1024;
		ROW_STEP = 16;
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_tails;
	unsigned *graph_vertices;
	unsigned *graph_edges;
	unsigned *graph_degrees;
	unsigned *tile_offsets;
	unsigned *tile_sizes;

	unsigned *graph_weights = nullptr;


#ifdef WEIGHTED
	input_weighted(
		filename, 
		graph_heads, 
		graph_tails, 
		graph_vertices,
		graph_edges,
		graph_degrees,
		graph_weights,
		tile_offsets,
		tile_sizes);
#else
	input(
		filename, 
		graph_heads, 
		graph_tails, 
		graph_vertices,
		graph_edges,
		graph_degrees,
		tile_offsets,
		tile_sizes);
#endif

	unsigned source = 0;
	// Map a vertex index to its new index: vertex_map[old] = new;
	unsigned *vertex_map = (unsigned *) malloc(NNODES * sizeof(unsigned));
#pragma omp parallel for num_threads(64)
	for (unsigned i = 0; i < NNODES; ++i) {
		vertex_map[i] = (unsigned) -1;
	}

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("Input finished: %s\n", filename);
	unsigned run_count = 7;
#else
	unsigned run_count = 7;
#endif
	//T_RATIO = 81;
	//T_RATIO = 60;
	//CHUNK_SIZE = 2048;
	//CHUNK_SIZE_DENSE = 32768;
	//SIZE_BUFFER_MAX = 512;
	//for (unsigned v = 5; v < 101; v += 5) {
	T_RATIO = 20;
	WORK_LOAD = 10;
	//CHUNK_SIZE_SPARSE = v;
	CHUNK_SIZE_DENSE = 1024;
	//CHUNK_SIZE_BLOCK = v;
	SIZE_BUFFER_MAX = 800;
	//printf("T_RATIO: %u\n", v);
	//SIZE_BUFFER_MAX = 1024;
	// BFS
	for (unsigned cz = 0; cz < 1; ++cz) {
	for (unsigned i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#ifndef ONEDEBUG
		//sleep(10);
#endif
		for (unsigned k = 0; k < 1; ++k) {
		BC(
			graph_heads, 
			graph_tails, 
			graph_vertices,
			graph_edges,
			graph_degrees,
			tile_offsets,
			tile_sizes,
			source,
			vertex_map);
		}
		//// Re-initializing
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	}
	//}
	fclose(time_out);
#ifdef WEIGHTED
	make_up_data_weighted(
			vertex_map,
			graph_heads,
			graph_tails,
			graph_weights,
			filename);
#else
	make_up_data(
			vertex_map,
			graph_heads,
			graph_tails,
			filename);
#endif

	// Free memory
	free(graph_heads);
	free(graph_tails);
	free(graph_vertices);
	free(graph_edges);
	free(graph_degrees);
	if (nullptr != graph_weights) {
		free(graph_weights);
	}
	free(tile_offsets);
	free(tile_sizes);
	free(vertex_map);

	return 0;
}
