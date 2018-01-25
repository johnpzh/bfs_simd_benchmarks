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
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;
using std::vector;

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

void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_tails, 
		unsigned *&graph_vertices,
		unsigned *&graph_edges,
		unsigned *&graph_degrees,
		unsigned *&tile_offsets,
		unsigned *&tile_sizes,
		unsigned *&graph_heads_reverse,
		unsigned *&graph_tails_reverse,
		unsigned *&graph_vertices_reverse,
		unsigned *&graph_edges_reverse,
		unsigned *&graph_degrees_reverse,
		unsigned *&tile_offsets_reverse,
		unsigned *&tile_sizes_reverse)
{
	//printf("data: %s\n", filename);
	string prefix = string(input_f) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
	string prefix_reverse = string(input_f) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH) + "_reverse";
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

	fname = prefix_reverse + "-offsets";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	tile_offsets_reverse = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		fscanf(fin, "%u", tile_offsets_reverse + i);
	}
	fclose(fin);
	tile_sizes_reverse = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			tile_sizes_reverse[i] = tile_offsets[i + 1] - tile_offsets[i];
		} else {
			tile_sizes_reverse[i] = NEDGES - tile_offsets[i];
		}
	}

	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_tails = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_vertices = (unsigned *) malloc(NNODES * sizeof(unsigned));
	graph_edges = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) malloc(NNODES * sizeof(unsigned));
	graph_heads_reverse = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_tails_reverse = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_vertices_reverse = (unsigned *) malloc(NNODES * sizeof(unsigned));
	graph_edges_reverse = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees_reverse = (unsigned *) malloc(NNODES * sizeof(unsigned));

	// FOr heads and tails
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
		h_graph_heads[index] = n1;
		h_graph_tails[index] = n2;
	}

}

	//For reverse graph CSR
	//vector< vector<unsigned>> graph_heads_reverse_bucket(NNODES);

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
		graph_edges[index] = n2;
#pragma omp critical
		{
		graph_heads_reverse_bucket[n2].push_back(n1);
		}
	}

	fclose(fin);
}
	// CSR
	unsigned edge_start = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		graph_vertices[i] = edge_start;
		edge_start += graph_degrees[i];
	}

	// Reverse CSR
	edge_start = 0;
	unsigned edge_i = 0;
	for (unsigned head = 0; head < NNODES; ++head) {
		graph_vertices_reverse[head] = edge_start;
		unsigned size = graph_heads_reverse_bucket[head].size();
		edge_start += size;
		for (unsigned tail_i = 0; tail_i < size; ++tail_i) {
			graph_edges_reverse[edge_i++] = graph_heads_reverse_bucket[head][tail_i];
		}
		graph_degrees_reverse[head] = size;
	}
	graph_heads_reverse_bucket.clear();
}

inline unsigned update_visited(
		int *h_graph_mask,
		int *h_graph_visited)
{
	unsigned count = 0;
#pragma omp parallel for reduction(+: count)
	for (unsigned i = 0; i < NNODES; ++i) {
		if (h_graph_mask[i]) {
			h_graph_visited[i] = 1;
			++count;
		}
	}
	return count;
}

inline void update_visited_reverse(
		int *h_graph_mask,
		int *h_graph_visited,
		float *dependencies,
		float *inverse_num_paths)
{
#pragma omp parallel for 
	for (unsigned i = 0; i < NNODES; ++i) {
		if (h_graph_mask[i]) {
			h_graph_visited[i] = 1;
			dependencies[i] += inverse_num_paths[i];
		}
	}
}

inline int *BFS_kernel(
		unsigned *graph_vertices,
		unsigned *graph_edges,
		unsigned *graph_degrees,
		int *h_graph_mask,
		int *h_graph_visited,
		unsigned *num_paths)
{
	int *new_frontier = (int *) calloc(NNODES, sizeof(int));
	//unsigned new_frontier_size = 0;

//#pragma omp parallel for schedule(dynamic, 512) reduction(+: new_frontier_size)
#pragma omp parallel for schedule(dynamic, 512)
	for(unsigned int head_id = 0; head_id < NNODES; head_id++ )
	{
		if (!h_graph_mask[head_id]) {
			continue;
		}
		//unsigned num_paths_head = num_paths[head_id];
		unsigned bound_edge_i = graph_vertices[head_id] + graph_degrees[head_id];
		for (unsigned edge_i = graph_vertices[head_id]; edge_i < bound_edge_i; ++edge_i) {
			unsigned tail_id = graph_edges[edge_i];
			if (h_graph_visited[tail_id]) {
				continue;
			}
			// Change in new_frontier
			if (!new_frontier[tail_id]) {
				new_frontier[tail_id] = 1;
			}
			// Update num_paths
			volatile unsigned old_val = num_paths[tail_id];
			volatile unsigned new_val = old_val + num_paths[head_id];
			while (!__sync_bool_compare_and_swap(num_paths + tail_id, old_val, new_val)) {
				old_val = num_paths[tail_id];
				new_val = old_val + num_paths[head_id];
			}
		}
	}

	return new_frontier;
}

void BFS_kernel_reverse(
			unsigned *graph_vertices_reverse,
			unsigned *graph_edges_reverse,
			unsigned *graph_degrees_reverse,
			int *h_graph_mask,
			int *h_graph_visited,
			unsigned *num_paths,
			float *dependencies)

{
#pragma omp parallel for schedule(dynamic, 512)
	for(unsigned int head_id = 0; head_id < NNODES; head_id++ )
	{
		if (!h_graph_mask[head_id]) {
			continue;
		}
		unsigned bound_edge_i = graph_vertices_reverse[head_id] + graph_degrees_reverse[head_id];
		for (unsigned edge_i = graph_vertices_reverse[head_id]; edge_i < bound_edge_i; ++edge_i) {
			unsigned tail_id = graph_edges_reverse[edge_i];
			if (h_graph_visited[tail_id]) {
				continue;
			}
			volatile float old_val;
			volatile float new_val;
			do {
				old_val = dependencies[tail_id];
				new_val = old_val + dependencies[head_id];
			} while (!__sync_bool_compare_and_swap(
											(int *) (dependencies + tail_id), 
											*((int *) &old_val), 
											*((int *) &new_val)));
		}
	}
}


void BC(
		unsigned *graph_heads, 
		unsigned *graph_tails, 
		unsigned *graph_vertices,
		unsigned *graph_edges,
		unsigned *graph_degrees,
		unsigned *graph_vertices_reverse,
		unsigned *graph_edges_reverse,
		unsigned *graph_degrees_reverse,
		const unsigned &source)
{
	omp_set_num_threads(NUM_THREADS);
	unsigned *num_paths = (unsigned *) calloc(NNODES, sizeof(unsigned));
	int *h_graph_visited = (int *) calloc(NNODES, sizeof(int));
	int *h_graph_mask = (int *) calloc(NNODES, sizeof(int));
	float *dependencies = (float *) calloc(NNODES, sizeof(float));
	vector<int *> frontiers;
	vector<unsigned> frontiers_sizes;
	unsigned frontier_size;

	num_paths[source] = 1;
	h_graph_visited[source] = 1;
	h_graph_mask[source] = 1;
	frontier_size = 1;

	double start_time = omp_get_wtime();
	frontiers.push_back(h_graph_mask);
	frontiers_sizes.push_back(frontier_size);

		
	// First phase
	while (0 != frontier_size) {
		int *new_frontier = BFS_kernel(
								graph_vertices,
								graph_edges,
								graph_degrees,
								h_graph_mask,
								h_graph_visited,
								num_paths);
		h_graph_mask = new_frontier;
		frontiers.push_back(h_graph_mask);
		frontier_size = update_visited(h_graph_mask, h_graph_visited);
		frontiers_sizes.push_back(frontier_size);
	}
	int level_count = frontiers.size() - 1;
	float *inverse_num_paths = (float *) malloc(NNODES * sizeof(float));
#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < NNODES; ++i) {
		if (num_paths[i] == 0) {
			inverse_num_paths[i] = 0.0;
		} else {
			inverse_num_paths[i] = 1/ (1.0 * num_paths[i]);
		}
	}
#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < NNODES; ++i) {
		h_graph_visited[i] = 0;
	}

	for (int lc = level_count - 1; lc >= 0; --lc) {
		h_graph_mask = frontiers[lc];
		//undate_visited();
		update_visited_reverse(
						h_graph_mask, 
						h_graph_visited,
						dependencies,
						inverse_num_paths);
		BFS_kernel_reverse(
					graph_vertices_reverse,
					graph_edges_reverse,
					graph_degrees_reverse,
					h_graph_mask,
					h_graph_visited,
					num_paths,
					dependencies);
	}

#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < NNODES; ++i) {
		if (inverse_num_paths[i] == 0.0) {
			dependencies[i] = 0.0;
		} else {
			dependencies[i] = (dependencies[i] - inverse_num_paths[i]) / inverse_num_paths[i];
		}
	}

	printf("%u %f\n", NUM_THREADS, omp_get_wtime() - start_time);
	
	// Free memory
	for (auto f = frontiers.begin(); f != frontiers.end(); ++f) {
		free(*f);
	}
	free(num_paths);
	free(h_graph_visited);
	free(dependencies);
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
		//filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		filename = "/sciclone/scr-mlt/zpeng01/pokec_combine/soc-pokec";
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

	unsigned *graph_heads_reverse;
	unsigned *graph_tails_reverse;
	unsigned *graph_vertices_reverse;
	unsigned *graph_edges_reverse;
	unsigned *graph_degrees_reverse;
	unsigned *tile_offsets_reverse;
	unsigned *tile_sizes_reverse;

	input(
		filename, 
		graph_heads, 
		graph_tails, 
		graph_vertices,
		graph_edges,
		graph_degrees,
		tile_offsets,
		tile_sizes,
		graph_heads_reverse;
		graph_tails_reverse;
		graph_vertices_reverse,
		graph_edges_reverse,
		graph_degrees_reverse,
		tile_offsets_reverse,
		tile_sizes_reverse);

	unsigned source = 0;

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("Input finished: %s\n", filename);
	unsigned run_count = 9;
#else
	unsigned run_count = 9;
#endif
	// BFS
	for (unsigned i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#ifndef ONEDEBUG
		//sleep(10);
#endif
		BC(
			graph_heads, 
			graph_tails, 
			graph_vertices,
			graph_edges,
			graph_degrees,
			graph_vertices_reverse,
			graph_edges_reverse,
			graph_degrees_reverse,
			source);
		//// Re-initializing
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);

	// Free memory
	free(graph_heads);
	free(graph_tails);
	free(graph_vertices);
	free(graph_edges);
	free(graph_degrees);
	free(tile_offsets);
	free(tile_sizes);
	free(graph_heads_reverse);
	free(graph_tails_reverse);
	free(graph_vertices_reverse);
	free(graph_edges_reverse);
	free(graph_degrees_reverse);
	free(tile_offsets_reverse);
	free(tile_sizes_reverse);

	return 0;
}
