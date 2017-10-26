#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>
#include <immintrin.h>

using std::string;
using std::to_string;

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH;
unsigned SIZE_BUFFER_MAX;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


void input(int argc, char** argv);
void BFS(\
		unsigned *h_graph_starts,\
		unsigned *h_graph_ends,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_cost,\
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		int *is_active_side,\
		int *is_updating_active_side,\
		unsigned side_length,\
		unsigned num_tiles,\
		unsigned row_step\
		);

inline void bfs_kernel(\
		unsigned *heads_buffer,
		unsigned *ends_buffer,
		const unsigned &size_buffer,
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_cost,\
		int *is_updating_active_side\
		)
{
	//for (unsigned edge_i = start_edge_i; edge_i < bound_edge_i; ) {}
	//for (unsigned edge_i = 0; edge_i < size_buffer;) {
	//	//unsigned head = h_graph_heads[edge_i];
	//	unsigned head = heads_buffer[edge_i];
	//	if (0 == h_graph_mask[head]) {
	//		++edge_i;
	//		continue;
	//	}
	//	while (heads_buffer[edge_i] == head) {
	//		//unsigned end = h_graph_ends[edge_i];
	//		unsigned end = ends_buffer[edge_i];
	//		if (!h_graph_visited[end]) {
	//			h_cost[end] = h_cost[head] + 1;
	//			h_updating_graph_mask[end] = 1;
	//			is_updating_active_side[end/TILE_WIDTH] = 1;
	//		}
	//		++edge_i;
	//	}
	//}

	unsigned edge_i;
	for (edge_i = 0; edge_i + NUM_P_INT <= size_buffer; edge_i += NUM_P_INT) {
		__m512i head_v = _mm512_load_epi32(heads_buffer + edge_i);
		__m512i active_flag_v = _mm512_i32gather_epi32(head_v, h_graph_mask, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			continue;
		}
		__m512i end_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, ends_buffer + edge_i);
		__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, end_v, h_graph_visited, sizeof(int));
		__mmask16 not_visited_m = _mm512_testn_epi32_mask(visited_flag_v, _mm512_set1_epi32(-1));
		if (!not_visited_m) {
			continue;
		}
		__m512i cost_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, h_cost, sizeof(int));
		__m512i cost_end_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, cost_head_v, _mm512_set1_epi32(1));
		_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, end_v, cost_end_v, sizeof(int));
		_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, end_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(end_v, TILE_WIDTH_v);
		_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}

	__m512i edge_i_v = _mm512_set_epi32(edge_i + 15, edge_i + 14, edge_i + 13, edge_i + 12,\
			edge_i + 11, edge_i + 10, edge_i + 9, edge_i + 8,\
			edge_i + 7, edge_i + 6, edge_i + 5, edge_i + 4,\
			edge_i + 3, edge_i + 2, edge_i + 1, edge_i);
	__m512i size_buffer_v = _mm512_set1_epi32(size_buffer);
	__mmask16 in_range_m = _mm512_cmplt_epi32_mask(edge_i_v, size_buffer_v);
	__m512i head_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, heads_buffer + edge_i);
	__m512i active_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, head_v, h_graph_mask, sizeof(int));
	__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
	if (!is_active_m) {
		return;
	}
	__m512i end_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, ends_buffer + edge_i);
	__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, end_v, h_graph_visited, sizeof(int));
	__mmask16 not_visited_m = _mm512_testn_epi32_mask(visited_flag_v, _mm512_set1_epi32(-1));
	if (!not_visited_m) {
		return;
	}
	__m512i cost_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, h_cost, sizeof(int));
	__m512i cost_end_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, cost_head_v, _mm512_set1_epi32(1));
	_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, end_v, cost_end_v, sizeof(int));
	_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, end_v, _mm512_set1_epi32(1), sizeof(int));
	__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
	__m512i side_id_v = _mm512_div_epi32(end_v, TILE_WIDTH_v);
	_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
}

inline void scheduler(\
		const unsigned &start_row_index,\
		const unsigned &bound_row_index,\
		unsigned *h_graph_heads,\
		unsigned *h_graph_ends,\
		unsigned *heads_buffer,
		unsigned *ends_buffer,
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_cost,\
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		const unsigned &side_length,\
		const unsigned &num_tiles,\
		int *is_updating_active_side\
		)
{
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned col_id = 0; col_id < side_length; ++col_id) {
		unsigned tid = omp_get_thread_num();
		unsigned *heads_buffer_base = heads_buffer + tid * SIZE_BUFFER_MAX;
		unsigned *ends_buffer_base = ends_buffer + tid * SIZE_BUFFER_MAX;
		unsigned size_buffer = 0;
		unsigned capacity = SIZE_BUFFER_MAX;
		for (unsigned row_id = start_row_index; row_id < bound_row_index; ++row_id) {
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
						memcpy(heads_buffer_base + size_buffer, h_graph_heads + edge_i, remain * sizeof(unsigned));
						memcpy(ends_buffer_base + size_buffer, h_graph_ends + edge_i, remain * sizeof(unsigned));
						edge_i += remain;
						capacity -= remain;
						size_buffer += remain;
						remain = 0;
					} else {
						// Fill the buffer to full
						memcpy(heads_buffer_base + size_buffer, h_graph_heads + edge_i, capacity * sizeof(unsigned));
						memcpy(ends_buffer_base + size_buffer, h_graph_ends + edge_i, capacity * sizeof(unsigned));
						edge_i += capacity;
						remain -= capacity;
						size_buffer += capacity;
						capacity = 0;
					}
				} else {
					// Process the full buffer
					bfs_kernel(
							heads_buffer_base,
							ends_buffer_base,
							size_buffer,
							h_graph_mask,\
							h_updating_graph_mask,\
							h_graph_visited,\
							h_cost,\
							is_updating_active_side\
							);
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}
			
			//bfs_kernel();
			//unsigned bound_edge_i;
			//if (num_tiles - 1 != tile_id) {
			//	bound_edge_i = tile_offsets[tile_id + 1];
			//} else {
			//	bound_edge_i = NEDGES;
			//}
			//bfs_kernel(\
			//		tile_offsets[tile_id],\
			//		bound_edge_i,\
			//		h_graph_heads,\
			//		h_graph_ends,\
			//		h_graph_mask,\
			//		h_updating_graph_mask,\
			//		h_graph_visited,\
			//		h_cost,\
			//		is_updating_active_side\
			//		);
		}
		// Process the remains in buffer
		bfs_kernel(
				heads_buffer_base,
				ends_buffer_base,
				size_buffer,
				h_graph_mask,\
				h_updating_graph_mask,\
				h_graph_visited,\
				h_cost,\
				is_updating_active_side\
				);
	}
}

void BFS(\
		unsigned *h_graph_heads,\
		unsigned *h_graph_ends,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_cost,\
		unsigned *tile_offsets,
		unsigned *tile_sizes,\
		int *is_active_side,\
		int *is_updating_active_side,\
		unsigned side_length,\
		unsigned num_tiles,\
		unsigned row_step
		)
{

	//printf("Start traversing the tree\n");
	omp_set_num_threads(NUM_THREADS);
	unsigned *heads_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned *ends_buffer = (unsigned *) _mm_malloc(sizeof(unsigned) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	double start_time = omp_get_wtime();
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;

//#pragma omp parallel for 
		//for(unsigned int nid = 0; nid < NNODES; nid++ )
		//{
		//	if (h_graph_mask[nid] == 1) {
		//		h_graph_mask[nid]=0;
		//		//int next_starting = h_graph_nodes[nid].starting + h_graph_nodes[nid].num_of_edges;
		//		//for(int i = h_graph_nodes[nid].starting; \
		//		//		i < next_starting; \
		//		//		i++)
		//		//{
		//		//	int id = h_graph_edges[i];
		//		//	if(!h_graph_visited[id])
		//		//	{
		//		//		h_cost[id]=h_cost[nid]+1;
		//		//		h_updating_graph_mask[id]=1;
		//		//	}
		//		//}
		//	}
		//}
		for (unsigned side_id = 0; side_id < side_length; ) {
			if (!is_active_side[side_id]) {
				++side_id;
				continue;
			}
			if (side_id + row_step < side_length) {
				scheduler(\
						side_id,\
						side_id + row_step,\
						h_graph_heads,\
						h_graph_ends,\
						heads_buffer,
						ends_buffer,
						h_graph_mask,\
						h_updating_graph_mask,\
						h_graph_visited,\
						h_cost,\
						tile_offsets,
						tile_sizes,
						side_length,\
						num_tiles,\
						is_updating_active_side
						);
				side_id += row_step;
			} else {
				scheduler(\
						side_id,\
						side_length,\
						h_graph_heads,\
						h_graph_ends,\
						heads_buffer,
						ends_buffer,
						h_graph_mask,\
						h_updating_graph_mask,\
						h_graph_visited,\
						h_cost,\
						tile_offsets,
						tile_sizes,
						side_length,\
						num_tiles,\
						is_updating_active_side
						);
				side_id = side_length;
			}
		}
		//for (unsigned side_id = 0; side_id < side_length; ++side_id) {
		//	if (!is_active_side[side_id]) {
		//		continue;
		//	}
		//	is_active_side[side_id] = 0;
		//	unsigned start_tile_id = side_id * side_length;
		//	unsigned bound_tile_id = start_tile_id + side_length;
		//	for (unsigned tile_id = start_tile_id; \
		//			tile_id < bound_tile_id;\
		//			++tile_id) {
		//		if (is_empty_tile[tile_id]) {
		//			continue;
		//		}
		//		unsigned bound_edge_i;
		//		if (num_tiles - 1 != tile_id) {
		//			bound_edge_i = tile_offsets[tile_id + 1];
		//		} else {
		//			bound_edge_i = NEDGES;
		//		}
		//		for (unsigned edge_i = tile_offsets[tile_id]; \
		//				edge_i < bound_edge_i; \
		//				) {
		//			unsigned head = h_graph_heads[edge_i];
		//			if (0 == h_graph_mask[head]) {
		//				edge_i++;
		//				continue;
		//			}
		//			int passed_count = 0;
		//			//unsigned i = edge_i;
		//			while (h_graph_heads[edge_i] == head) {
		//				unsigned end = h_graph_ends[edge_i];
		//				if (!h_graph_visited[end]) {
		//					h_cost[end] = h_cost[head] + 1;
		//					h_updating_graph_mask[end] = 1;
		//					is_updating_active_side[end/TILE_WIDTH] = 1;
		//				}
		//				edge_i++;
		//			}
		//		}

		//	}
		//}
#pragma omp parallel for
		//for(unsigned int nid=0; nid< NNODES ; nid++ )
		//{
		//	if (h_updating_graph_mask[nid] == 1) {
		//		h_graph_mask[nid]=1;
		//		h_graph_visited[nid]=1;
		//		stop = false;
		//		h_updating_graph_mask[nid]=0;
		//	}
		//}
		for (unsigned side_id = 0; side_id < side_length; ++side_id) {
			if (!is_updating_active_side[side_id]) {
				is_active_side[side_id] = 0;
				continue;
			}
			is_updating_active_side[side_id] = 0;
			is_active_side[side_id] = 1;
			stop = false;
			unsigned bound_vertex_id;
			if (side_length - 1 != side_id) {
				bound_vertex_id = side_id * TILE_WIDTH + TILE_WIDTH;
			} else {
				bound_vertex_id = NNODES;
			}
			//for (unsigned i = 0; i < TILE_WIDTH; ++i) {
			//	unsigned vertex_id = i + side_id * TILE_WIDTH;
			//	if (vertex_id == NNODES) {
			//		break;
			//	}
			//	if (1 == h_updating_graph_mask[vertex_id]) {
			//		h_updating_graph_mask[vertex_id] = 0;
			//		h_graph_mask[vertex_id] = 1;
			//		h_graph_visited[vertex_id] = 1;
			//	} else {
			//		h_graph_mask[vertex_id] = 0;
			//	}
			//}
			unsigned vertex_id;
			for (vertex_id = side_id * TILE_WIDTH; \
					vertex_id + NUM_P_INT <= bound_vertex_id; \
					vertex_id += NUM_P_INT) {
				__m512i updating_flag_v = _mm512_loadu_si512(h_updating_graph_mask + vertex_id);
				__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
				if (!is_updating_m) {
					continue;
				}
				_mm512_mask_storeu_epi32(h_updating_graph_mask + vertex_id, is_updating_m, _mm512_set1_epi32(0));
				_mm512_storeu_si512(h_graph_mask + vertex_id, updating_flag_v);
				_mm512_mask_storeu_epi32(h_graph_visited + vertex_id, is_updating_m, _mm512_set1_epi32(1));
			}

			__m512i vertex_id_v = _mm512_set_epi32(
											vertex_id + 15, vertex_id + 14, vertex_id + 13, vertex_id + 12,\
											vertex_id + 11, vertex_id + 10, vertex_id + 9, vertex_id + 8,\
											vertex_id + 7, vertex_id + 6, vertex_id + 5, vertex_id + 4,\
											vertex_id + 3, vertex_id + 2, vertex_id + 1, vertex_id);
			__m512i bound_vertex_id_v = _mm512_set1_epi32(bound_vertex_id);
			__mmask16 in_range_m = _mm512_cmplt_epi32_mask(vertex_id_v, bound_vertex_id_v);
			__m512i updating_flag_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), in_range_m, h_updating_graph_mask + vertex_id);
			__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
			if (!is_updating_m) {
				continue;
			}
			_mm512_mask_storeu_epi32(h_updating_graph_mask + vertex_id, is_updating_m, _mm512_set1_epi32(0));
			_mm512_storeu_si512(h_graph_mask + vertex_id, updating_flag_v);
			_mm512_mask_storeu_epi32(h_graph_visited + vertex_id, is_updating_m, _mm512_set1_epi32(1));
		}
	}
	while(!stop);
	double end_time = omp_get_wtime();
	printf("%d %lf\n", NUM_THREADS, (end_time - start_time));
	_mm_free(heads_buffer);
	_mm_free(ends_buffer);
}
///////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph
///////////////////////////////////////////////////////////////////////////////
void input( int argc, char** argv) 
{
	int num_of_indices;
	char *input_f;
	
	if(argc < 3){
		input_f = "/home/zpeng/benchmarks/data/pokec/coo_tiled_bak/soc-pokec";
		TILE_WIDTH = 1024;
	} else {
		input_f = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
	}

	/////////////////////////////////////////////////////////////////////
	// Input real dataset
	/////////////////////////////////////////////////////////////////////
	//string prefix = string(input_f) + "_untiled";
	string prefix = string(input_f) + "_coo-tiled-" + to_string(TILE_WIDTH);
	//string prefix = string(input_f) + "_col-16-coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	fclose(fin);
	unsigned side_length;
	if (NNODES % TILE_WIDTH) {
		side_length = NNODES / TILE_WIDTH + 1;
	} else {
		side_length = NNODES / TILE_WIDTH;
	}
	unsigned num_tiles = side_length * side_length;
	// Read tile Offsets
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
	for (unsigned i = 0; i < num_tiles; ++i) {
		if (num_tiles - 1 != i) {
			tile_sizes[i] = tile_offsets[i + 1] - tile_offsets[i];
		} else {
			tile_sizes[i] = NEDGES - tile_offsets[i];
		}
	}
	unsigned *h_graph_starts = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
	unsigned *h_graph_ends = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
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
		h_graph_starts[index] = n1;
		h_graph_ends[index] = n2;
	}

}
	// End Input real dataset
	/////////////////////////////////////////////////////////////////////

	int *h_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_graph_visited = (int*) malloc(sizeof(int)*NNODES);
	int* h_cost = (int*) malloc(sizeof(int)*NNODES);
	int *is_active_side = (int *) malloc(sizeof(int) * side_length);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * side_length);
	unsigned source = 0;

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("Input finished: %s\n", input_f);
	unsigned run_count = 9;
#else
	unsigned run_count = 9;
#endif
	// BFS
	for (unsigned row_step = 1; row_step < 10000; row_step *= 2) {
	printf("===========================\n");
	printf("row_step: %u\n", row_step);
	for (unsigned i = 4; i < 16; ++i) {
	SIZE_BUFFER_MAX = (unsigned) pow(2, i);
	printf("SIZE_BUFFER_MAX: %u\n", SIZE_BUFFER_MAX);
	//unsigned row_step = 256;
	//SIZE_BUFFER_MAX = 4096;

	if (side_length < row_step) {
		fprintf(stderr, "Error: row step is too large.\n");
		exit(3);
	}
	for (unsigned i = 6; i < run_count; ++i) {
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
		memset(is_active_side, 0, sizeof(int) * side_length);
		is_active_side[0] = 1;
		memset(is_updating_active_side, 0, sizeof(int) * side_length);

		BFS(\
			h_graph_starts,\
			h_graph_ends,\
			h_graph_mask,\
			h_updating_graph_mask,\
			h_graph_visited,\
			h_cost,\
			tile_offsets,
			tile_sizes,\
			is_active_side,\
			is_updating_active_side,\
			side_length,\
			num_tiles,\
			row_step\
			);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
#ifdef ONEDEBUG
		printf("Thread %u finished.\n", NUM_THREADS);
#endif
	}
	}
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
		fprintf(stderr, "Error: cannot open file %s.\n", file_name.c_str());
		exit(1);
	}
	unsigned bound_index;
	if (tid != NUM_THREADS - 1) {
		bound_index = offset + num_lines;
	} else {
		bound_index = NNODES;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
	}

	fclose(fpo);
}
#endif

	// cleanup memory
	free( h_graph_starts);
	free( h_graph_ends);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	free( tile_offsets);
	free( tile_sizes);
	free( is_active_side);
	free( is_updating_active_side);
}
///////////////////////////////////////////////////////////////////////////////
// Main Program
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	start = omp_get_wtime();
	input( argc, argv);
}

