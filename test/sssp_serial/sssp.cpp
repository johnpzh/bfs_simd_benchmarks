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
		unsigned *&graph_ends, 
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
	graph_ends = (unsigned *) malloc(nedges * sizeof(unsigned));
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
		graph_ends[index] = n2;
		graph_weights[index] = wt;
	}
	fclose(fin);
}
}

inline void sssp_kernel_weighted(
				unsigned *graph_heads, 
				unsigned *graph_ends, 
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
		unsigned end = graph_ends[edge_i];
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
					unsigned *graph_ends, 
					unsigned *graph_weights,
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
	const unsigned bound_row_index = start_row_index + tile_step;
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
			sssp_kernel_weighted(
				graph_heads, 
				graph_ends, 
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

void sssp_weighted(
		unsigned *graph_heads, 
		unsigned *graph_ends, 
		unsigned *graph_weights,
		unsigned *tile_offsets,
		int *graph_active, 
		int *graph_updating_active,
		int *is_active_side,
		int *is_updating_active_side,
		int *is_empty_tile,
		int *dists)
{
	omp_set_num_threads(NUM_THREADS);
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
			scheduler_weighted(
				graph_heads, 
				graph_ends, 
				graph_weights,
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
			scheduler_weighted(
					graph_heads, 
					graph_ends, 
					graph_weights,
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
}
// End Weighted Graph
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
// Unweighted Graph
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
	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		if (0 == graph_active[head]) {
			continue;
		}
		unsigned end = graph_ends[edge_i];
		if (-1 == dists[end] || dists[head] + 1 < dists[end]) {
			dists[end] = dists[head] + 1;
			graph_updating_active[end] = 1;
			is_updating_active_side[end/TILE_WIDTH] = 1;
		}
	}
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
				graph_ends, 
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
				graph_ends, 
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
					graph_ends, 
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
	unsigned *graph_ends;
	unsigned *graph_weights = nullptr;
	unsigned *tile_offsets;
	int *is_empty_tile;
	//unsigned *nneibor;
//#ifdef ONESERIAL
//	input_serial("data.txt", graph_heads, graph_ends);
//#else
//	input(
//		filename, 
//		graph_heads, 
//		graph_ends, 
//		tile_offsets,
//		is_empty_tile);
//#endif
	if (is_weighted_graph) {
		input_weighted(
				filename, 
				graph_heads, 
				graph_ends, 
				graph_weights,
				tile_offsets,
				is_empty_tile);
	} else {
		input(
				filename, 
				graph_heads, 
				graph_ends, 
				tile_offsets,
				is_empty_tile);
	}

	// SSSP
	int *distances = (int *) malloc(nnodes * sizeof(int));
	int *graph_active = (int *) malloc(nnodes * sizeof(int));
	int *graph_updating_active = (int *) malloc(nnodes * sizeof(int));
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
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
		memset(distances, -1, nnodes * sizeof(int));
		distances[source] = 0;
		memset(graph_active, 0, nnodes * sizeof(int));
		graph_active[source] = 1;
		memset(graph_updating_active, 0, nnodes * sizeof(int));
		memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
		is_active_side[source/TILE_WIDTH] = 1;
		memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);

		//sleep(10);
		if (is_weighted_graph) {
			sssp_weighted(
				graph_heads, 
				graph_ends, 
				graph_weights,
				tile_offsets,
				graph_active, 
				graph_updating_active,
				is_active_side,
				is_updating_active_side,
				is_empty_tile,
				distances);
		} else {
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
	free(graph_ends);
	if (nullptr != graph_weights) {
		free(graph_weights);
	}
	free(tile_offsets);
	free(is_empty_tile);
	free(distances);
	free(graph_active);
	free(graph_updating_active);
	free(is_active_side);
	free(is_updating_active_side);

	return 0;
}
