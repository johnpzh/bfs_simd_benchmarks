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
#include <getopt.h>
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

////////////////////////////////////////////////////////////
// Weighted Graph version
void input_weighted(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
		unsigned *&graph_weights,
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
	graph_tails = (unsigned *) malloc(nedges * sizeof(unsigned));
	graph_weights = (unsigned *) malloc(nedges * sizeof(unsigned));
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
		fscanf(fin, "%u%u\n", &nnodes, &nedges);
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
}

////////////////////////////////////////////////////////////
// Dense, Weighted Graph
inline void sssp_kernel_weighted(
				unsigned *graph_heads, 
				unsigned *graph_tails, 
				unsigned *graph_weights,
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
		unsigned end = graph_tails[edge_i];
		unsigned new_dist = dists[head] + graph_weights[edge_i];
		if (-1 == dists[end] || new_dist < dists[end]) {
			dists[end] = new_dist;
			graph_updating_active[end] = 1;
			is_updating_active_side[end/TILE_WIDTH] = 1;
		}
	}
}

inline void scheduler_weighted(
					unsigned *graph_heads, 
					unsigned *graph_tails, 
					unsigned *graph_weights,
					unsigned *tile_offsets,
					unsigned *tile_sizes,
					int *graph_active, 
					int *graph_updating_active,
					int *is_active_side,
					int *is_updating_active_side,
					int *dists, 
					const unsigned &start_row_index,
					const unsigned &tile_step)
{
	const unsigned bound_row_index = start_row_index + tile_step;
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
				bound_edge_i = nedges;
			}
			sssp_kernel_weighted(
				graph_heads, 
				graph_tails, 
				graph_weights,
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
inline void BFS_dense_weighted(
					unsigned *graph_heads,
					unsigned *graph_tails,
					unsigned *graph_weights,
					unsigned *tile_offsets,
					unsigned *tile_sizes,
					int *h_graph_mask,
					int *is_updating_graph_mask,
					int *is_active_side,
					int *is_updating_active_side,
					unsigned *dists)
{
	unsigned remainder = SIDE_LENGTH % ROW_STEP;
	unsigned bound_side_id = SIDE_LENGTH - remainder;
	for (unsigned side_id = 0; side_id < bound_side_id; side_id += ROW_STEP) {
		scheduler_weighted(
				graph_heads, 
				graph_tails, 
				graph_weights,
				tile_offsets,
				tile_sizes,
				h_graph_mask, 
				is_updating_graph_mask,
				is_active_side,
				is_updating_active_side,
				dists, 
				side_id,
				ROW_STEP);
		//side_id += ROW_STEP;
	}
	if (remainder > 0) {
		scheduler_weighted(
				graph_heads, 
				graph_tails, 
				graph_weights,
				tile_offsets,
				tile_sizes,
				h_graph_mask, 
				is_updating_graph_mask,
				is_active_side,
				is_updating_active_side,
				dists, 
				bound_side_id,
				remainder);
	}

}
// End Dense, Weighte Graph
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// Sparse, Weighted Graph

inline void to_dense(
					h_graph_queue,
					frontier_size,
					h_graph_mask,
					is_active_side)
{
	memset(h_graph_mask, 0, NNODES * sizeof(int));
	memset(is_active_side, 0, SIDE_LENGTH * sizeof(int));
#pragma omp parallel for
	for (unsigned i = 0; i< frontier_sizes; ++i) {
		unsigned vertex_id = h_graph_queue[i];
		h_graph_mask[vertex_id] = 1;
		is_active_side[vertex_id / TILE_WIDTH] = 1;
	}
}
inline unsigned sparse_update_weighted(
				unsigned *h_graph_queue,
				const unsigned &queue_size,
				unsigned graph_degrees)
{
	unsigned out_degree = 0;
#pragma omp parallel for reduction(+: out_degree)
	for (unsigned i = 0; i < queue_size; ++i) {
		unsigned vertex_id = h_graph_queue[i];
		out_degree += graph_degrees[vertex_id];
	}
}

inline unsigned *BFS_kernel_sparse_weighted(
				unsigned *graph_vertices,
				unsigned *graph_edges,
				unsigned *graph_degrees,
				//int *h_graph_visited,
				unsigned *h_graph_queue,
				unsigned &queue_size,
				//unsigned *num_paths,
				unsigned *dists)
{
	// From h_graph_queue, get the degrees (para_for)
	unsigned *degrees = (unsigned *) malloc(sizeof(unsigned) *  queue_size);
	unsigned new_queue_size = 0;
//#pragma omp parallel for schedule(dynamic) reduction(+: new_queue_size)
#pragma omp parallel for reduction(+: new_queue_size)
	for (unsigned i = 0; i < queue_size; ++i) {
		degrees[i] = graph_degrees[h_graph_queue[i]];
		new_queue_size += degrees[i];
	}
	if (0 == new_queue_size) {
		free(degrees);
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
	unsigned *new_frontier_tmp = (unsigned *) malloc(sizeof(unsigned) * new_queue_size);
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
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
			if (new_dist < dits[end]) {
				volatile unsigned old_val = dists[end];
				volatile unsigned new_val = new_dist;
				bool dist_updated = __sync_bool_compare_and_swap(dists + end, old_val, new_val);
				if (dist_updated) {
					new_frontier_tmp[frontier_i] = end;
				} else {
					new_frontier_tmp[frontier_i] = (unsigned) -1;
				}
			} else {
				new_frontier_tmp[frontier_i] = (unsigned) -1;
			}
			////////////////
			//if (0 == h_graph_visited[end]) {
			//	volatile unsigned old_val;
			//	volatile unsigned new_val;
			//	do {
			//		old_val = num_paths[end];
			//		new_val = old_val + num_paths[start];
			//	} while (!__sync_bool_compare_and_swap(num_paths + end, old_val, new_val));
			//	if (old_val == 0.0) {
			//		new_frontier_tmp[offset + k] = end;
			//	} else {
			//		new_frontier_tmp[offset + k] = (unsigned) -1;
			//	}
			//} else {
			//	new_frontier_tmp[offset + k] = (unsigned) -1;
			//}
			///////////////
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
		free(degrees);
		free(new_frontier_tmp);
		if (nums_in_blocks) {
			free(nums_in_blocks);
		}
		queue_size = 0;
		return nullptr;
	}

	// Get the final new frontier
	unsigned *new_frontier = (unsigned *) malloc(sizeof(unsigned) * new_queue_size);
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
		for (unsigned i = offset; i < bound; ++i) {
			new_frontier[i] = new_frontier_tmp[base++];
		}
	}
	} else {
		unsigned base = 0;
		for (unsigned i = 0; i < new_queue_size; ++i) {
			new_frontier[i] = new_frontier_tmp[base++];
		}
	}

	// Return the results
	free(degrees);
	free(new_frontier_tmp);
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
				//int *h_graph_visited,
				//unsigned *num_paths
				unsigned *dists)
{
	return BFS_kernel_sparse_weighted(
				graph_vertices,
				graph_edges,
				graph_degrees,
				//h_graph_visited,
				h_graph_queue,
				queue_size,
				//num_paths,
				dists);
}
// End Sparse, Weighted Graph
////////////////////////////////////////////////////////////

void sssp_weighted(
		unsigned *graph_heads, 
		unsigned *graph_tails, 
		unsigned *graph_weights,
		unsigned *tile_offsets,
		//int *graph_active, 
		//int *graph_updating_active,
		//int *is_active_side,
		//int *is_updating_active_side,
		int *is_empty_tile,
		//int *dists,
		const unsigned &source)
{
	omp_set_num_threads(NUM_THREADS);

	unsigned *dists = (unsigned *) malloc(nnodes * sizeof(int));
	//int *graph_active = (int *) malloc(nnodes * sizeof(int));
	//int *graph_updating_active = (int *) malloc(nnodes * sizeof(int));
	int *h_graph_mask = (int *) malloc(nnodes * sizeof(int));
	int *is_updating_graph_mask = (int *) malloc(nnodes * sizeof(int));
	int *new_mask = nullptr;
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned *h_graph_queue = nullptr;
	unsigned frontier_size;
	unsigned out_degree;
	bool last_is_dense;

	memset(dists, -1, nnodes * sizeof(int));
	dists[source] = 0;
	memset(graph_active, 0, nnodes * sizeof(int));
	graph_active[source] = 1;
	memset(graph_updating_active, 0, nnodes * sizeof(int));
	memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
	is_active_side[source/TILE_WIDTH] = 1;
	memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);

	double start_time = omp_get_wtime();

	// Fisrt is Sparse
	frontier_size = 1;
	h_graph_queue = (unsigned *) malloc(frontier_size *sizeof(unsigned));
	h_graph_queue[0] = source;
	unsigned *new_queue = BFS_sparse_weighted(
								h_graph_queue,
								frontier_size,
								graph_vertices,
								graph_edges,
								graph_degrees,
								//h_graph_visited,
								//num_paths
								dists);
	free(h_graph_queue);
	h_graph_queue = new_queue;
	last_is_dense = false;
	// Get the sum of the number of active nodes and their out degrees
	out_degree =  sparse_update_weighted(
							h_graph_queue,
							frontier_size,
							graph_degrees);

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
					tile_offsets,
					tile_sizes,
					h_graph_mask,
					is_updating_graph_mask,
					is_active_side,
					is_updating_active_side,
					dists);
			last_is_dense = true;
		} else {
			// Sparse
			if (last_is_dense) {
				//HERE
			}
		}
		///////////////////////////////////////
		//unsigned remainder = SIDE_LENGTH % ROW_STEP;
		//unsigned bound_side_id = SIDE_LENGTH - remainder;
		//for (unsigned side_id = 0; side_id < bound_side_id; side_id += ROW_STEP) {
		//	//if (!is_active_side[side_id]) {
		//	//	++side_id;
		//	//	continue;
		//	//}
		//	scheduler_weighted(
		//		graph_heads, 
		//		graph_tails, 
		//		graph_weights,
		//		tile_offsets,
		//		graph_active, 
		//		graph_updating_active,
		//		is_active_side,
		//		is_updating_active_side,
		//		is_empty_tile,
		//		dists, 
		//		side_id,
		//		ROW_STEP);
		//	//side_id += ROW_STEP;
		//}
		//if (remainder > 0) {
		//	scheduler_weighted(
		//			graph_heads, 
		//			graph_tails, 
		//			graph_weights,
		//			tile_offsets,
		//			graph_active, 
		//			graph_updating_active,
		//			is_active_side,
		//			is_updating_active_side,
		//			is_empty_tile,
		//			dists, 
		//			bound_side_id,
		//			remainder);
		//}
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

	free(dists);
	//free(graph_active);
	//free(graph_updating_active);
	free(h_graph_mask);
	free(is_updating_graph_mask);
	free(is_active_side);
	free(is_updating_active_side);
	free(h_graph_queue);
}
// End Weighted Graph
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// Unweighted Graph
void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
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
	graph_tails = (unsigned *) malloc(nedges * sizeof(unsigned));
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
		graph_tails[index] = n2;
	}
	fclose(fin);
}
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
				unsigned *graph_tails, 
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
		unsigned end = graph_tails[edge_i];
		if (-1 == dists[end] || dists[head] + 1 < dists[end]) {
			dists[end] = dists[head] + 1;
			graph_updating_active[end] = 1;
			is_updating_active_side[end/TILE_WIDTH] = 1;
		}
	}
}

inline void scheduler(
					unsigned *graph_heads, 
					unsigned *graph_tails, 
					unsigned *tile_offsets,
					int *graph_active, 
					int *graph_updating_active,
					int *is_active_side,
					int *is_updating_active_side,
					int *is_empty_tile,
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
			if (is_empty_tile[tile_id] || !is_active_side[row_id]) {
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
				graph_tails, 
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
		unsigned *graph_tails, 
		unsigned *tile_offsets,
		//int *graph_active, 
		//int *graph_updating_active,
		//int *is_active_side,
		//int *is_updating_active_side,
		int *is_empty_tile,
		//int *dists,
		const unsigned source)
{
	omp_set_num_threads(NUM_THREADS);

	int *dists = (int *) malloc(nnodes * sizeof(int));
	int *graph_active = (int *) malloc(nnodes * sizeof(int));
	int *graph_updating_active = (int *) malloc(nnodes * sizeof(int));
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);

	memset(dists, -1, nnodes * sizeof(int));
	dists[source] = 0;
	memset(graph_active, 0, nnodes * sizeof(int));
	graph_active[source] = 1;
	memset(graph_updating_active, 0, nnodes * sizeof(int));
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
				graph_active, 
				graph_updating_active,
				is_active_side,
				is_updating_active_side,
				is_empty_tile,
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
					graph_active, 
					graph_updating_active,
					is_active_side,
					is_updating_active_side,
					is_empty_tile,
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

	free(dists);
	free(graph_active);
	free(graph_updating_active);
	free(is_active_side);
	free(is_updating_active_side);
}
// End Unweighted Graph
////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) 
{
	int is_weighted_graph = 0;
	// Process the options
	start = omp_get_wtime();
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
	int *is_empty_tile;
	//unsigned *nneibor;
//#ifdef ONESERIAL
//	input_serial("data.txt", graph_heads, graph_tails);
//#else
//	input(
//		filename, 
//		graph_heads, 
//		graph_tails, 
//		tile_offsets,
//		is_empty_tile);
//#endif
	if (is_weighted_graph) {
		input_weighted(
				filename, 
				graph_heads, 
				graph_tails, 
				graph_weights,
				tile_offsets,
				is_empty_tile);
	} else {
		input(
				filename, 
				graph_heads, 
				graph_tails, 
				tile_offsets,
				is_empty_tile);
	}

	// SSSP
	//int *distances = (int *) malloc(nnodes * sizeof(int));
	//int *graph_active = (int *) malloc(nnodes * sizeof(int));
	//int *graph_updating_active = (int *) malloc(nnodes * sizeof(int));
	//int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	//int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned source = 0;
	
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("SSSP starts...\n");
	unsigned run_count = 9;
#else
	unsigned run_count = 9;
#endif
	for (unsigned i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		//memset(distances, -1, nnodes * sizeof(int));
		//distances[source] = 0;
		//memset(graph_active, 0, nnodes * sizeof(int));
		//graph_active[source] = 1;
		//memset(graph_updating_active, 0, nnodes * sizeof(int));
		//memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
		//is_active_side[source/TILE_WIDTH] = 1;
		//memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);

		//sleep(10);
		if (is_weighted_graph) {
			sssp_weighted(
				graph_heads, 
				graph_tails, 
				graph_weights,
				tile_offsets,
				//graph_active, 
				//graph_updating_active,
				//is_active_side,
				//is_updating_active_side,
				is_empty_tile,
				//distances,
				source);
		} else {
			sssp(
				graph_heads, 
				graph_tails, 
				tile_offsets,
				//graph_active, 
				//graph_updating_active,
				//is_active_side,
				//is_updating_active_side,
				is_empty_tile,
				//distances,
				source);
		}
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);
#ifdef ONEDEBUG
	print(distances);
#endif

	// Free memory
	free(graph_heads);
	free(graph_tails);
	if (nullptr != graph_weights) {
		free(graph_weights);
	}
	free(tile_offsets);
	free(is_empty_tile);
	//free(distances);
	//free(graph_active);
	//free(graph_updating_active);
	//free(is_active_side);
	//free(is_updating_active_side);

	return 0;
}
