#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>
#include "../../include/peg_count.h"

using std::string;
using std::to_string;

struct Vertex {
	unsigned *out_neighbors;
	unsigned out_degree;

	unsigned get_out_neighbor(unsigned i) {
		return out_neighbors[i];
	}
	unsigned get_out_degree() {
		return out_degree;
	}
};

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH;
unsigned SIDE_LENGTH;
unsigned NUM_TILES;
unsigned ROW_STEP;
unsigned CHUNK_SIZE;
unsigned T_RATIO;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

//////////////////////////////////////////////////////////////////
// Dense (bottom-up)
void to_dense(
		int *h_graph_mask,
		int *is_active_side,
		unsigned *frontier,
		const unsigned &frontier_size)
{
	memset(h_graph_mask, 0, NNODES * sizeof(int));
	memset(is_active_side, 0, SIDE_LENGTH * sizeof(int));
#pragma omp parallel for
	for (unsigned i = 0; i < frontier_size; ++i) {
		unsigned vertex_id = frontier[i];
		h_graph_mask[vertex_id] = 1;
		is_active_side[vertex_id / TILE_WIDTH] = 1;
	}
}

inline void update_dense(
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
			// Think about this bug. How did you find it? And, more importantly,
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
				frontier_size++;
				out_degree += graph_degrees[vertex_id];

				bot_access_counter.count(vertex_id);
			} else {
				h_graph_mask[vertex_id] = 0;
			}
		}
	}
	_frontier_size = frontier_size;
	_out_degree = out_degree;
}
inline void bfs_kernel_dense(
		const unsigned &start_edge_i,
		const unsigned &bound_edge_i,
		unsigned *graph_heads,
		unsigned *graph_tails,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		//int *h_graph_visited,
		unsigned *h_graph_parents,
		int *h_cost,
		int *is_updating_active_side)
{
	for (unsigned edge_i = start_edge_i; edge_i < bound_edge_i; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		if (0 == h_graph_mask[head]) {
			//++edge_i;
			continue;
		}
		unsigned end = graph_tails[edge_i];


		//if (!h_graph_visited[end]) {}
		if ((unsigned) -1 == h_graph_parents[end]) {
			h_cost[end] = h_cost[head] + 1;
			h_updating_graph_mask[end] = 1;
			is_updating_active_side[end/TILE_WIDTH] = 1;
			h_graph_parents[end] = head; // addition
		}
	}
}
inline void scheduler_dense(
		const unsigned &start_row_index,
		const unsigned &tile_step,
		unsigned *graph_heads,
		unsigned *graph_tails,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		//int *h_graph_visited,
		unsigned *h_graph_parents,
		int *h_cost,
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		int *is_active_side,
		int *is_updating_active_side)
{
	unsigned start_tile_id = start_row_index * SIDE_LENGTH;
	//unsigned bound_row_id = start_row_index + tile_step;
	unsigned end_tile_id = start_tile_id + tile_step * SIDE_LENGTH;
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
		unsigned bound_tile_id = tile_index + tile_step;
		for (unsigned tile_id = tile_index; tile_id < bound_tile_id; ++tile_id) {
			unsigned row_id = (tile_id - start_tile_id) % tile_step + start_row_index;
			if (0 == tile_sizes[tile_id] || !is_active_side[row_id]) {
				continue;
			}
			// Kernel
			unsigned bound_edge_i;
			if (NUM_TILES - 1 != tile_id) {
				bound_edge_i = tile_offsets[tile_id + 1];
			} else {
				bound_edge_i = NEDGES;
			}
			bfs_kernel_dense(
					tile_offsets[tile_id],
					bound_edge_i,
					graph_heads,
					graph_tails,
					h_graph_mask,
					h_updating_graph_mask,
					//h_graph_visited,
					h_graph_parents,
					h_cost,
					is_updating_active_side
					);
		}

	}
}

void BFS_dense(
		unsigned *graph_heads,
		unsigned *graph_tails,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		//int *h_graph_visited,
		unsigned *h_graph_parents,
		int *h_cost,
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		int *is_active_side,
		int *is_updating_active_side)
{

	unsigned side_id;
	for (side_id = 0; side_id + ROW_STEP <= SIDE_LENGTH; side_id += ROW_STEP) {
		scheduler_dense(
				//side_id * SIDE_LENGTH,\
				//(side_id + ROW_STEP) * SIDE_LENGTH,
				side_id,
				ROW_STEP,
				graph_heads,
				graph_tails,
				h_graph_mask,
				h_updating_graph_mask,
				//h_graph_visited,
				h_graph_parents,
				h_cost,
				tile_offsets,
				tile_sizes,
				is_active_side,
				is_updating_active_side);
	}
	scheduler_dense(
			//side_id * SIDE_LENGTH,\
			//NUM_TILES,
			side_id,
			SIDE_LENGTH - side_id,
			graph_heads,
			graph_tails,
			h_graph_mask,
			h_updating_graph_mask,
			//h_graph_visited,
			h_graph_parents,
			h_cost,
			tile_offsets,
			tile_sizes,
			is_active_side,
			is_updating_active_side);

}
// End Dense (bottom-up)
///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
// Sparse (top-down)
unsigned *to_sparse(
		unsigned *frontier,
		const unsigned &frontier_size,
		int *h_graph_mask)
{
	unsigned *new_frontier = (unsigned *) malloc(frontier_size * sizeof(unsigned));

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

inline unsigned update_sparse(
				unsigned *h_graph_queue,
				const unsigned &queue_size,
				unsigned *graph_degrees,
				unsigned *h_graph_parents,
				int *h_cost)
{
	unsigned out_degree = 0;
#pragma omp parallel for reduction(+: out_degree)
	for (unsigned i = 0; i < queue_size; ++i) {
		unsigned end = h_graph_queue[i];
		unsigned start = h_graph_parents[end];
		h_cost[end] = h_cost[start] + 1;
		out_degree += graph_degrees[end];

		bot_access_counter.count(end);
	}

	return out_degree;
}

double offset_time1 = 0;
double offset_time2 = 0;
double degree_time = 0;
double frontier_tmp_time = 0;
double refine_time = 0;
double arrange_time = 0;
double run_time = 0;
unsigned *BFS_kernel_sparse(
				//unsigned *graph_vertices,
				//Vertex *graph_vertices_info,
				//unsigned *graph_edges,
				unsigned *graph_vertices,
				unsigned *graph_edges,
				unsigned *graph_degrees,
				unsigned *h_graph_parents,
				unsigned *frontier,
				unsigned &frontier_size)
{
	// From frontier, get the degrees (para_for)
	double time_now = omp_get_wtime(); 
	unsigned *degrees = (unsigned *) malloc(sizeof(unsigned) *  frontier_size);
	//Vertex *frontier_vertices = (Vertex *) malloc(sizeof(Vertex) * frontier_size);
	unsigned new_frontier_size = 0;
//#pragma omp parallel for schedule(dynamic) reduction(+: new_frontier_size)
#pragma omp parallel for reduction(+: new_frontier_size)
	for (unsigned i = 0; i < frontier_size; ++i) {
		degrees[i] = graph_degrees[frontier[i]];
		new_frontier_size += degrees[i];
	}
	//for (unsigned i = 0; i < frontier_size; ++i) {
	//	unsigned start = frontier[i];
	//	Vertex v = graph_vertices_info[start];
	//	degrees[i] = v.get_out_degree();
	//	new_frontier_size += degrees[i];
	//	frontier_vertices[i] = v;
	//}
	if (0 == new_frontier_size) {
		free(degrees);
		frontier_size = 0;
		//frontier = nullptr;
		return nullptr;
	}
	degree_time += omp_get_wtime() - time_now;

	// From degrees, get the offset (stored in degrees) (block_para_for)
	// TODO: blocked parallel for
	//unsigned *offsets = (unsigned *) malloc(sizeof(unsigned) * frontier_size);
	time_now = omp_get_wtime();
	unsigned offset_sum = 0;
	for (unsigned i = 0; i < frontier_size; ++i) {
		unsigned tmp = degrees[i];
		degrees[i] = offset_sum;
		offset_sum += tmp;
	}
	offset_time1 += omp_get_wtime() - time_now;
	//offsets[0] = 0;
	//for (unsigned i = 1; i < frontier_size; ++i) {
	//	offsets[i] = offsets[i - 1] + degrees[i - 1];
	//}

	// From offset, get active vertices (para_for)
	time_now = omp_get_wtime();
	unsigned *new_frontier_tmp = (unsigned *) malloc(sizeof(unsigned) * new_frontier_size);
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
	for (unsigned i = 0; i < frontier_size; ++i) {
		unsigned start = frontier[i];
		unsigned offset = degrees[i];
		unsigned out_degree = graph_degrees[start];
		unsigned base = graph_vertices[start];
		for (unsigned k = 0; k < out_degree; ++k) {
			unsigned end = graph_edges[base + k];



			if ((unsigned) -1 == h_graph_parents[end]) {
				bool unvisited = __sync_bool_compare_and_swap(h_graph_parents + end, (unsigned) -1, start); //update h_graph_parents
				if (unvisited) {
					new_frontier_tmp[offset + k] = end;
				} else {
					new_frontier_tmp[offset + k] = (unsigned) -1;
				}


			} else {
				new_frontier_tmp[offset + k] = (unsigned) -1;
			}
		}
	}
	frontier_tmp_time += omp_get_wtime() - time_now;


	// Refine active vertices, removing visited and redundant (block_para_for)
	//unsigned block_size = new_frontier_size / NUM_THREADS;
	time_now = omp_get_wtime();
	unsigned block_size = 1024 * 2;
	//unsigned num_blocks = new_frontier_size % block_size == 0 ? new_frontier_size/block_size : new_frontier_size/block_size + 1;
	unsigned num_blocks = (new_frontier_size - 1)/block_size + 1;

	unsigned *nums_in_blocks = NULL;
	if (num_blocks > 1) {
	//unsigned *nums_in_blocks = (unsigned *) malloc(sizeof(unsigned) * NUM_THREADS);
	nums_in_blocks = (unsigned *) malloc(sizeof(unsigned) * num_blocks);
	unsigned new_frontier_size_tmp = 0;
//#pragma omp parallel for schedule(dynamic) reduction(+: new_frontier_size_tmp)
#pragma omp parallel for reduction(+: new_frontier_size_tmp)
	for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
		unsigned offset = block_i * block_size;
		unsigned bound;
		if (num_blocks - 1 != block_i) {
			bound = offset + block_size;
		} else {
			bound = new_frontier_size;
		}
		//unsigned size = 0;
		unsigned base = offset;
		for (unsigned end_i = offset; end_i < bound; ++end_i) {
			if ((unsigned) - 1 != new_frontier_tmp[end_i]) {
				new_frontier_tmp[base++] = new_frontier_tmp[end_i];
			}
		}
		nums_in_blocks[block_i] = base - offset;
		new_frontier_size_tmp += nums_in_blocks[block_i];
	}
	new_frontier_size = new_frontier_size_tmp;
	} else {
		unsigned base = 0;
		for (unsigned i = 0; i < new_frontier_size; ++i) {
			if ((unsigned) -1 != new_frontier_tmp[i]) {
				new_frontier_tmp[base++] = new_frontier_tmp[i];
			}
		}
		new_frontier_size = base;
	}
	refine_time += omp_get_wtime() - time_now;
	
	if (0 == new_frontier_size) {
		//free(offsets);
		//free(frontier_vertices);
		free(degrees);
		free(new_frontier_tmp);
		if (nums_in_blocks) {
			free(nums_in_blocks);
		}
		frontier_size = 0;
		return nullptr;
	}

	// Get the final new frontier
	time_now = omp_get_wtime();
	unsigned *new_frontier = (unsigned *) malloc(sizeof(unsigned) * new_frontier_size);
	if (num_blocks > 1) {
	//TODO: blocked parallel for
	double time_now = omp_get_wtime();
	offset_sum = 0;
	for (unsigned i = 0; i < num_blocks; ++i) {
		unsigned tmp = nums_in_blocks[i];
		nums_in_blocks[i] = offset_sum;
		offset_sum += tmp;
		//offsets_b[i] = offsets_b[i - 1] + nums_in_blocks[i - 1];
	}
	offset_time2 += omp_get_wtime() - time_now;
//#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for
	for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
		//unsigned offset = offsets_b[block_i];
		unsigned offset = nums_in_blocks[block_i];
		unsigned bound;
		if (num_blocks - 1 != block_i) {
			bound = nums_in_blocks[block_i + 1];
		} else {
			bound = new_frontier_size;
		}
		//unsigned bound = offset + nums_in_blocks[block_i];
		unsigned base = block_i * block_size;
		for (unsigned i = offset; i < bound; ++i) {
			new_frontier[i] = new_frontier_tmp[base++];
		}
	}
	} else {
		unsigned base = 0;
		for (unsigned i = 0; i < new_frontier_size; ++i) {
			new_frontier[i] = new_frontier_tmp[base++];
		}
	}
	arrange_time += omp_get_wtime() - time_now;

	// Return the results
	//free(frontier_vertices);
	free(degrees);
	free(new_frontier_tmp);
	if (nums_in_blocks) {
		free(nums_in_blocks);
	}
	frontier_size = new_frontier_size;
	return new_frontier;
}
unsigned *BFS_sparse(
		//unsigned *graph_vertices,
		//Vertex *graph_vertices_info,
		unsigned *frontier,
		unsigned *graph_vertices,
		unsigned *graph_edges,
		unsigned *graph_degrees,
		unsigned *h_graph_parents,
		//unsigned *graph_edges,
		//unsigned *graph_degrees,
		//const unsigned &source,
		unsigned &frontier_size)
		//int *h_cost)
{

	//return BFS_kernel_sparse(
	//		//graph_vertices,
	//		graph_vertices_info,
	//		//graph_edges,
	//		//graph_degrees,
	//		h_graph_parents,
	//		frontier,
	//		frontier_size);
	return BFS_kernel_sparse(
				//unsigned *graph_vertices,
				//Vertex *graph_vertices_info,
				//unsigned *graph_edges,
				graph_vertices,
				graph_edges,
				graph_degrees,
				h_graph_parents,
				frontier,
				frontier_size);
	//printf("@614\n");
}
// End Sparse (top-down)
///////////////////////////////////////////////////////////////////////////////


void graph_prepare(
		unsigned *graph_vertices,
		unsigned *graph_edges,
		unsigned *graph_heads,
		unsigned *graph_tails,
		unsigned *graph_degrees,
		unsigned *tile_offsets,
		unsigned *tile_sizes,
		const unsigned &source,
		int max_hop)
{
	//int max_hop = 3;
	int hop = 0;

	// Set up
	omp_set_num_threads(NUM_THREADS);
	int *h_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_cost = (int*) malloc(sizeof(int)*NNODES);
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned *h_graph_parents = (unsigned *) malloc(sizeof(unsigned) * NNODES);
	
	memset(h_graph_mask, 0, sizeof(int)*NNODES);
	memset(h_updating_graph_mask, 0, sizeof(int)*NNODES);
#pragma omp parallel for num_threads(64)
	for (unsigned j = 0; j < NNODES; ++j) {
		h_cost[j] = -1;
	}
	h_cost[source] = 0;
	memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
	memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);
#pragma omp parallel for num_threads(64)
	for (unsigned j = 0; j < NNODES; ++j) {
		h_graph_parents[j] = (unsigned) -1; // means unvisited yet
	}
	h_graph_parents[source] = source;

	// The first time, running the Sparse
	unsigned frontier_size = 1;
	unsigned *frontier = (unsigned *) malloc(sizeof(unsigned) * frontier_size);
	frontier[0] = source;
	double start_time = omp_get_wtime();
	unsigned *new_frontier = BFS_sparse(
								frontier,
								graph_vertices,
								graph_edges,
								graph_degrees,
								h_graph_parents,
								frontier_size);
	free(frontier);
	frontier = new_frontier;
	//printf("%d %lf\n", CHUNK_SIZE, run_time = (end_time - start_time));

	// When update the parents, get the sum of the number of active nodes and their out degree.
	bool last_is_dense = false;
	unsigned out_degree = 0;
	out_degree = update_sparse(
							frontier,
							frontier_size,
							graph_degrees,
							h_graph_parents,
							h_cost);
	++hop;
	// According the sum, determine to run Sparse or Dense, and then change the last_is_dense.
	//unsigned bfs_threshold = NEDGES / 20; // Determined according to Ligra
	unsigned bfs_threshold = NEDGES / T_RATIO; // Determined according to Ligra
	while (hop <= max_hop) {
		if (frontier_size + out_degree > bfs_threshold) {
			if (!last_is_dense) {
				to_dense(
					h_graph_mask, 
					is_active_side, 
					frontier, 
					frontier_size);
			}
			BFS_dense(
					graph_heads,
					graph_tails,
					h_graph_mask,
					h_updating_graph_mask,
					//h_graph_visited,
					h_graph_parents,
					h_cost,
					tile_offsets,
					tile_sizes,
					is_active_side,
					is_updating_active_side);
			last_is_dense = true;
			// Update the parents, and get the out_degree again
			update_dense(
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
				new_frontier = to_sparse(
					frontier,
					frontier_size,
					h_graph_mask);
				free(frontier);
				frontier = new_frontier;
			}
			new_frontier = BFS_sparse(
								frontier,
								graph_vertices,
								graph_edges,
								graph_degrees,
								h_graph_parents,
								frontier_size);
			free(frontier);
			frontier = new_frontier;
			last_is_dense = false;
			// Update the parents, and get the out_degree again
			if (0 == frontier_size) {
				break;
			}
			out_degree = update_sparse(
								frontier,
								frontier_size,
								graph_degrees,
								h_graph_parents,
								h_cost);
		}
		++hop;
	}
	double end_time = omp_get_wtime();
	fprintf(stderr, "%d %lf\n", NUM_THREADS, run_time = (end_time - start_time));
	free(frontier);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_parents);
	free( h_cost);
	free( is_active_side);
	free( is_updating_active_side);
}

///////////////////////////////////////////////////////////////////////////////
// Input data and then apply BFS
///////////////////////////////////////////////////////////////////////////////
void graph_input(
			char *input_f,
			unsigned *&graph_heads,
			unsigned *&graph_tails,
			unsigned *&tile_offsets,
			unsigned *&tile_sizes,
			unsigned *&graph_vertices,
			unsigned *&graph_edges,
			unsigned *&graph_degrees)
{
	/////////////////////////////////////////////////////////////////////
	// Input real dataset
	/////////////////////////////////////////////////////////////////////
	//string prefix = string(input_f) + "_untiled";
	//string prefix = string(input_f) + "_coo-tiled-" + to_string(TILE_WIDTH);
	string prefix = string(input_f) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
	//string prefix = string(input_f) + "_col-2-coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
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
	graph_heads = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
	graph_tails = (unsigned *) malloc(sizeof(unsigned) * NEDGES);

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


	// For Sparse
	prefix = string(input_f) + "_untiled";
	graph_edges = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
	graph_degrees = (unsigned *) malloc(sizeof(unsigned) * NNODES);

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
	graph_vertices = (unsigned *) malloc(sizeof(unsigned) * NNODES);
	unsigned edge_start = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		graph_vertices[i] = edge_start;
		edge_start += graph_degrees[i];
	}
	// End Input real dataset
	/////////////////////////////////////////////////////////////////////


}
//void input( int argc, char** argv) 
//{
//	char *input_f;
//	//ROW_STEP = 2;
//	
//	if(argc < 4){
//		//input_f = "/home/zpeng/benchmarks/data/pokec_combine/soc-pokec";
//		input_f = "/sciclone/scr-mlt/zpeng01/pokec_combine/soc-pokec";
//		TILE_WIDTH = 1024;
//		ROW_STEP = 16;
//	} else {
//		input_f = argv[1];
//		TILE_WIDTH = strtoul(argv[2], NULL, 0);
//		ROW_STEP = strtoul(argv[3], NULL, 0);
//	}
//
//	/////////////////////////////////////////////////////////////////////
//	// Input real dataset
//	/////////////////////////////////////////////////////////////////////
//	//string prefix = string(input_f) + "_untiled";
//	//string prefix = string(input_f) + "_coo-tiled-" + to_string(TILE_WIDTH);
//	string prefix = string(input_f) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
//	//string prefix = string(input_f) + "_col-2-coo-tiled-" + to_string(TILE_WIDTH);
//	string fname = prefix + "-0";
//	FILE *fin = fopen(fname.c_str(), "r");
//	fscanf(fin, "%u %u", &NNODES, &NEDGES);
//	fclose(fin);
//	if (NNODES % TILE_WIDTH) {
//		SIDE_LENGTH = NNODES / TILE_WIDTH + 1;
//	} else {
//		SIDE_LENGTH = NNODES / TILE_WIDTH;
//	}
//	NUM_TILES = SIDE_LENGTH * SIDE_LENGTH;
//	// Read tile Offsets
//	fname = prefix + "-offsets";
//	fin = fopen(fname.c_str(), "r");
//	if (!fin) {
//		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
//		exit(1);
//	}
//	unsigned *tile_offsets = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
//	for (unsigned i = 0; i < NUM_TILES; ++i) {
//		fscanf(fin, "%u", tile_offsets + i);
//	}
//	fclose(fin);
//	unsigned *graph_heads = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
//	unsigned *graph_tails = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
//	int *is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
//	memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);
//
//	NUM_THREADS = 64;
//	unsigned edge_bound = NEDGES / NUM_THREADS;
//#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
//{
//	unsigned tid = omp_get_thread_num();
//	unsigned offset = tid * edge_bound;
//	fname = prefix + "-" + to_string(tid);
//	fin = fopen(fname.c_str(), "r");
//	if (!fin) {
//		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
//		exit(1);
//	}
//	if (0 == tid) {
//		fscanf(fin, "%u %u", &NNODES, &NEDGES);
//	}
//	unsigned bound_index;
//	if (NUM_THREADS - 1 != tid) {
//		bound_index = offset + edge_bound;
//	} else {
//		bound_index = NEDGES;
//	}
//	for (unsigned index = offset; index < bound_index; ++index) {
//		unsigned n1;
//		unsigned n2;
//		fscanf(fin, "%u %u", &n1, &n2);
//		n1--;
//		n2--;
//		graph_heads[index] = n1;
//		graph_tails[index] = n2;
//	}
//	fclose(fin);
//
//}
//
//
//	// For Sparse
//	prefix = string(input_f) + "_untiled";
//	unsigned *graph_edges = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
//	unsigned *graph_degrees = (unsigned *) malloc(sizeof(unsigned) * NNODES);
//	memset(graph_degrees, 0, sizeof(unsigned) * NNODES);
//
//	// Read degrees
//	fname = prefix + "-nneibor";
//	fin = fopen(fname.c_str(), "r");
//	if (!fin) {
//		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
//		exit(1);
//	}
//	for (unsigned i = 0; i < NNODES; ++i) {
//		fscanf(fin, "%u", graph_degrees + i);
//	}
//	fclose(fin);
//
//	NUM_THREADS = 64;
//	edge_bound = NEDGES / NUM_THREADS;
//#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
//{
//	unsigned tid = omp_get_thread_num();
//	unsigned offset = tid * edge_bound;
//	fname = prefix + "-" + to_string(tid);
//	fin = fopen(fname.c_str(), "r");
//	if (!fin) {
//		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
//		exit(1);
//	}
//	if (0 == tid) {
//		fscanf(fin, "%u %u", &NNODES, &NEDGES);
//	}
//	unsigned bound_index;
//	if (NUM_THREADS - 1 != tid) {
//		bound_index = offset + edge_bound;
//	} else {
//		bound_index = NEDGES;
//	}
//	for (unsigned index = offset; index < bound_index; ++index) {
//		unsigned n1;
//		unsigned n2;
//		fscanf(fin, "%u %u", &n1, &n2);
//		n1--;
//		n2--;
//		graph_edges[index] = n2;
//	}
//	fclose(fin);
//
//}
//	// CSR
//	//Vertex *graph_vertices_info = (Vertex *) malloc(sizeof(Vertex) * NNODES);
//	unsigned *graph_vertices = (unsigned *) malloc(sizeof(unsigned) * NNODES);
//	unsigned edge_start = 0;
//	for (unsigned i = 0; i < NNODES; ++i) {
//		graph_vertices[i] = edge_start;
//		edge_start += graph_degrees[i];
//		//graph_vertices_info[i].out_neighbors = graph_edges + edge_start;
//		//graph_vertices_info[i].out_neighbors = graph_edges + edge_start;
//		//graph_vertices_info[i].out_degree = graph_degrees[i];
//	}
//	//memcpy(graph_edges, graph_tails, sizeof(unsigned) * NEDGES);
//	//free(graph_heads);
//	//free(graph_tails);
//	// End Input real dataset
//	/////////////////////////////////////////////////////////////////////
//
//	int *h_graph_mask = (int*) malloc(sizeof(int)*NNODES);
//	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*NNODES);
//	int *h_cost = (int*) malloc(sizeof(int)*NNODES);
//	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
//	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
//	unsigned *h_graph_parents = (unsigned *) malloc(sizeof(unsigned) * NNODES);
//	unsigned source = 0;
//
//#ifdef ONEDEBUG
//	printf("Input finished: %s\n", input_f);
//	unsigned run_count = 9;
//#else
//	unsigned run_count = 9;
//#endif
//	// BFS
//	T_RATIO = 100;
//	CHUNK_SIZE = 2048;
//	for (unsigned i = 6; i < run_count; ++i) {
//		NUM_THREADS = (unsigned) pow(2, i);
//#ifndef ONEDEBUG
//		//sleep(10);
//#endif
//		// Re-initializing
//		for (unsigned k = 0; k < 3; ++k) {
//		memset(h_graph_mask, 0, sizeof(int)*NNODES);
//		//h_graph_mask[source] = 1;
//		memset(h_updating_graph_mask, 0, sizeof(int)*NNODES);
//#pragma omp parallel for num_threads(64)
//		for (unsigned j = 0; j < NNODES; ++j) {
//			h_cost[j] = -1;
//		}
//		h_cost[source] = 0;
//		memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
//		//is_active_side[0] = 1;
//		memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);
//#pragma omp parallel for num_threads(64)
//		for (unsigned j = 0; j < NNODES; ++j) {
//			h_graph_parents[j] = (unsigned) -1; // means unvisited yet
//		}
//		h_graph_parents[source] = source;
//
//		graph_prepare(
//				//unsigned *graph_vertices,
//				//graph_vertices_info,
//				graph_vertices,
//				graph_edges,
//				graph_heads,
//				graph_tails, 
//				graph_degrees,
//				h_graph_parents, // h_graph_visited
//				tile_offsets,
//				source,
//				h_graph_mask,
//				h_updating_graph_mask,
//				h_cost,
//				is_empty_tile,
//				is_active_side,
//				is_updating_active_side);
//	}
//	}
//	//}
//
//	//Store the result into a file
//
//#ifdef ONEDEBUG
//	NUM_THREADS = 64;
//	omp_set_num_threads(NUM_THREADS);
//	unsigned num_lines = NNODES / NUM_THREADS;
//#pragma omp parallel
//{
//	unsigned tid = omp_get_thread_num();
//	unsigned offset = tid * num_lines;
//	string file_prefix = "path/path";
//	string file_name = file_prefix + to_string(tid) + ".txt";
//	FILE *fpo = fopen(file_name.c_str(), "w");
//	if (!fpo) {
//		fprintf(stderr, "Error: cannot open file %s.\n", file_name.c_str());
//		exit(1);
//	}
//	unsigned bound_index;
//	if (tid != NUM_THREADS - 1) {
//		bound_index = offset + num_lines;
//	} else {
//		bound_index = NNODES;
//	}
//	for (unsigned index = offset; index < bound_index; ++index) {
//		fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
//	}
//
//	fclose(fpo);
//}
//#endif
//
//	// cleanup memory
//	free( graph_heads);
//	free( graph_tails);
//	free( graph_edges);
//	free( graph_degrees);
//	free( graph_vertices);
//	free( h_graph_mask);
//	free( h_updating_graph_mask);
//	free( h_graph_parents);
//	free( h_cost);
//	free( tile_offsets);
//	free( is_empty_tile);
//	free( is_active_side);
//	free( is_updating_active_side);
//}
///////////////////////////////////////////////////////////////////////////////
// Main Program
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	int max_hop;
	int max_count;
	char *input_f;
	
	if(argc > 5){
		input_f = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
		ROW_STEP = strtoul(argv[3], NULL, 0);
		max_hop = strtoul(argv[4], NULL, 0);
		max_count = strtoul(argv[5], NULL, 0);
	} else {
		puts("Usage: ./bfs <data> <tile_size> <stripe_length> <max_hop> <max_count>");
		exit(1);
	}

	// Input
	unsigned *graph_heads = nullptr;
	unsigned *graph_tails = nullptr;
	unsigned *tile_offsets = nullptr;
	unsigned *tile_sizes = nullptr;
	unsigned *graph_vertices = nullptr;
	unsigned *graph_edges = nullptr;
	unsigned *graph_degrees = nullptr;

	//input( argc, argv);
	graph_input(
			input_f,
			graph_heads,
			graph_tails,
			tile_offsets,
			tile_sizes,
			graph_vertices,
			graph_edges,
			graph_degrees);

	bot_access_counter.init(NNODES);
	NUM_THREADS = 64;
	for (unsigned source = 0; source < NNODES; source += NNODES/max_count) {
		//unsigned source = 0;
		// BFS
		//T_RATIO = 100;
		T_RATIO = 20;
		CHUNK_SIZE = 2048;
		// Re-initializing

		graph_prepare(
				graph_vertices,
				graph_edges,
				graph_heads,
				graph_tails, 
				graph_degrees,
				tile_offsets,
				tile_sizes,
				source,
				max_hop);

	}
	bot_access_counter.print();
	// cleanup memory
	free( graph_heads);
	free( graph_tails);
	free( graph_edges);
	free( graph_degrees);
	free( graph_vertices);
	free( tile_offsets);
	free( tile_sizes);
}

//int main( int argc, char** argv) 
//{
//	start = omp_get_wtime();
//	input( argc, argv);
//}

