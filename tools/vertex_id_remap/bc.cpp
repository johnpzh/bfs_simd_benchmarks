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

inline void bfs_kernel_dense(
		const unsigned &start_edge_i,
		const unsigned &bound_edge_i,
		unsigned *h_graph_heads,
		unsigned *h_graph_tails,
		unsigned *h_graph_mask,
		unsigned *h_updating_graph_mask,
		int *h_graph_visited,
		//unsigned *h_graph_parents,
		//int *h_cost,
		int *is_updating_active_side,
		unsigned *num_paths)
{
	for (unsigned edge_i = start_edge_i; edge_i < bound_edge_i; ++edge_i) {
		unsigned head = h_graph_heads[edge_i];
		if (0 == h_graph_mask[head]) {
			//++edge_i;
			continue;
		}
		unsigned end = h_graph_tails[edge_i];
		//if ((unsigned) -1 == h_graph_parents[end]) {
		//	h_cost[end] = h_cost[head] + 1;
		//	h_updating_graph_mask[end] = 1;
		//	is_updating_active_side[end/TILE_WIDTH] = 1;
		//	h_graph_parents[end] = head; // addition
		//}
		if (0 == h_graph_visited[end]) {
			volatile unsigned old_val;
			volatile unsigned new_val;
			do {
				old_val = num_paths[end];
				new_val = old_val + num_paths[head];
			} while (!__sync_bool_compare_and_swap(num_paths + end, old_val, new_val));
			if (old_val == 0.0) {
				h_updating_graph_mask[end] = 1;
				is_updating_active_side[end/TILE_WIDTH] = 1;
			}
		}
	}
}
inline void scheduler_dense(
		const unsigned &start_row_index,
		const unsigned &tile_step,
		unsigned *h_graph_heads,
		unsigned *h_graph_tails,
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
	//unsigned bound_row_id = start_row_index + tile_step;
	unsigned end_tile_id = start_tile_id + tile_step * SIDE_LENGTH;
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
		unsigned bound_tile_id = tile_index + tile_step;
		for (unsigned tile_id = tile_index; tile_id < bound_tile_id; ++tile_id) {
			unsigned row_id = (tile_id - start_tile_id) % tile_step + start_row_index;
			//if (is_empty_tile[tile_id] || !is_active_side[row_id]) {
			//	continue;
			//}
			if (!tile_sizes[tile_id] || !is_active_side[row_id]) {
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
					h_graph_heads,
					h_graph_tails,
					h_graph_mask,
					h_updating_graph_mask,
					h_graph_visited,
					//h_graph_parents,
					//h_cost,
					is_updating_active_side,
					num_paths);
		}

	}
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
	unsigned *new_mask = (unsigned *) calloc(NNODES, sizeof(unsigned));
	unsigned side_id;
	for (side_id = 0; side_id + ROW_STEP <= SIDE_LENGTH; side_id += ROW_STEP) {
		scheduler_dense(
				//side_id * SIDE_LENGTH,\
				//(side_id + ROW_STEP) * SIDE_LENGTH,
				side_id,
				ROW_STEP,
				h_graph_heads,
				h_graph_tails,
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
			SIDE_LENGTH - side_id,
			h_graph_heads,
			h_graph_tails,
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

	return new_mask;
}


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
	unsigned top_index = 0;

	omp_set_num_threads(NUM_THREADS);
	unsigned *num_paths = (unsigned *) calloc(NNODES, sizeof(unsigned));
	int *h_graph_visited = (int *) calloc(NNODES, sizeof(int));
	int *is_active_side = (int *) calloc(SIDE_LENGTH, sizeof(int));
	int *is_updating_active_side = (int *) calloc(SIDE_LENGTH, sizeof(int));
	unsigned *h_graph_mask = nullptr;
	float *dependencies = (float *) calloc(NNODES, sizeof(float));
	vector<unsigned *> frontiers;
	vector<unsigned> frontier_sizes;
	vector<bool> is_dense_frontier;
	unsigned frontier_size;

	num_paths[source] = 1;
	// First is the Sparse
	frontier_size = 1;
	h_graph_visited[source] = 1;
	unsigned *h_graph_queue = (unsigned *) malloc(frontier_size * sizeof(unsigned));
	h_graph_queue[0] = source;
	frontiers.push_back(h_graph_queue);
	is_dense_frontier.push_back(false);
	frontier_sizes.push_back(1);

	vertex_map[source] = top_index++;

	double start_time = omp_get_wtime();
	// First Phase
	// According the sum, determing to run Sparse or Dense, and then change the last_is_dense.
	unsigned bfs_threshold = NEDGES / T_RATIO;
	while (true) {
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
			h_graph_mask = new_mask;
			frontiers.push_back(h_graph_mask);

			add_mask2map(
					vertex_map,
					top_index,
					h_graph_mask);

		// Update h_graph_visited; Get the sum again.
			frontier_size = 0;
#pragma omp parallel for reduction(+: frontier_size)
			for (unsigned side_id = 0; side_id < SIDE_LENGTH; ++side_id) {
				if (!is_updating_active_side[side_id]) {
					is_active_side[side_id] = 0;
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
					if (1 == h_graph_mask[vertex_id]) {
						frontier_size++;
						h_graph_visited[vertex_id] = 1;
					}
				}
			}
			frontier_sizes.push_back(frontier_size);
			if (0 == frontier_size) {
				break;
			}
	}
	add_remainder2map(
		vertex_map,
		top_index);


	// Free memory
	for (auto f = frontiers.begin(); f != frontiers.end(); ++f) {
		free(*f);
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
	free(num_paths);
	free(h_graph_visited);
	free(dependencies);
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
