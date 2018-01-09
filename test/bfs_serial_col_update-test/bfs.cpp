#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>

using std::string;
using std::to_string;

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH;
unsigned SIDE_LENGTH;
unsigned NUM_TILES;
unsigned ROW_STEP;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

inline void bfs_kernel(\
		const unsigned &start_edge_i,\
		const unsigned &bound_edge_i,\
		unsigned *h_graph_heads,\
		unsigned *h_graph_ends,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_cost,\
		int *is_updating_active_side\
		)
{
	for (unsigned edge_i = start_edge_i; edge_i < bound_edge_i; ) {
		unsigned head = h_graph_heads[edge_i];
		if (0 == h_graph_mask[head]) {
			++edge_i;
			continue;
		}
		while (h_graph_heads[edge_i] == head) {
			unsigned end = h_graph_ends[edge_i];
			if (!h_graph_visited[end]) {
				h_cost[end] = h_cost[head] + 1;
				h_updating_graph_mask[end] = 1;
				is_updating_active_side[end/TILE_WIDTH] = 1;
			}
			++edge_i;
		}
	}
}

//inline void scheduler(\
//		const unsigned &start_tile_id,\
//		const unsigned &end_tile_id,\
//		const unsigned &tile_step,
//		unsigned *h_graph_heads,\
//		unsigned *h_graph_ends,\
//		int *h_graph_mask,\
//		int *h_updating_graph_mask,\
//		int *h_graph_visited,\
//		int *h_cost,\
//		unsigned *tile_offsets,
//		int *is_empty_tile,\
//		int *is_active_side,
//		int *is_updating_active_side,
//		const unsigned &group_id
//		)
//{
//#pragma omp parallel for schedule(dynamic, 1)
//	//for (unsigned col_id = 0; col_id < SIDE_LENGTH; ++col_id) {}
//	for (unsigned tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
//		unsigned bound_tile_id = tile_index + tile_step;
//		//for (unsigned row_id = start_row_index; row_id < bound_row_index; ++row_id) {}
//		//	unsigned tile_id = row_id * SIDE_LENGTH + col_id;
//		for (unsigned tile_id = tile_index; tile_id < bound_tile_id; ++tile_id) {
//			//if (is_empty_tile[tile_id] || !is_active_side[tile_id/SIDE_LENGTH]) {
//			//	continue;
//			//}
//			if (is_empty_tile[tile_id]) {
//				continue;
//			}
//			unsigned side_id;
//			if (group_id != GROUP_MAX - 1) {
//				unsigned col_id = tile_id/tile_step;
//				side_id = tile_id - col_id * tile_step;
//			} else {
//				unsigned tmp_id = tile_id - group_id * GROUP_SIZE;
//				unsigned col_id = tmp_id/tile_step;
//				unsigned tmp_side_id = tmp_id - col_id * tile_step;
//				side_id = tmp_side_id + group_id * ROW_STEP;
//			}
//			if (!is_active_side[side_id]) {
//				continue;
//			}
//			// Kernel
//			unsigned bound_edge_i;
//			if (NUM_TILES - 1 != tile_id) {
//				bound_edge_i = tile_offsets[tile_id + 1];
//			} else {
//				bound_edge_i = NEDGES;
//			}
//			bfs_kernel(\
//					tile_offsets[tile_id],\
//					bound_edge_i,\
//					h_graph_heads,\
//					h_graph_ends,\
//					h_graph_mask,\
//					h_updating_graph_mask,\
//					h_graph_visited,\
//					h_cost,\
//					is_updating_active_side\
//					);
//			//for (unsigned edge_i = tile_offsets[tile_id]; edge_i < bound_edge_i; ) {
//			//	unsigned head = h_graph_heads[edge_i];
//			//	if (0 == h_graph_mask[head]) {
//			//		++edge_i;
//			//		continue;
//			//	}
//			//	while (h_graph_heads[edge_i] == head) {
//			//		unsigned end = h_graph_ends[edge_i];
//			//		if (!h_graph_visited[end]) {
//			//			h_cost[end] = h_cost[head] + 1;
//			//			h_updating_graph_mask[end] = 1;
//			//			is_updating_active_side[end/TILE_WIDTH] = 1;
//			//		}
//			//		++edge_i;
//			//	}
//			//}
//		}
//	}
//}
//inline void scheduler(\
//		const unsigned &start_row_index,\
//		const unsigned &tile_step,
//		unsigned *h_graph_heads,\
//		unsigned *h_graph_ends,\
//		int *h_graph_mask,\
//		int *h_updating_graph_mask,\
//		int *h_graph_visited,\
//		int *h_cost,\
//		unsigned *tile_offsets,
//		int *is_empty_tile,\
//		int *is_active_side,
//		int *is_updating_active_side)
//{
//	unsigned base_id = start_row_index * SIDE_LENGTH;
//	unsigned bound_row_id = start_row_index + tile_step;
//#pragma omp parallel for schedule(dynamic, 1)
//	for (unsigned col_id = 0; col_id < SIDE_LENGTH; ++col_id) {
//		unsigned stripe_start_id = base_id + col_id * tile_step;
//		for (unsigned row_id = start_row_index; row_id < bound_row_id; ++row_id) {
//			unsigned tile_id = stripe_start_id + row_id - start_row_index;
//			if (is_empty_tile[tile_id] || !is_active_side[row_id]) {
//				continue;
//			}
//			// Kernel
//			unsigned bound_edge_i;
//			if (NUM_TILES - 1 != tile_id) {
//				bound_edge_i = tile_offsets[tile_id + 1];
//			} else {
//				bound_edge_i = NEDGES;
//			}
//			bfs_kernel(\
//					tile_offsets[tile_id],\
//					bound_edge_i,\
//					h_graph_heads,\
//					h_graph_ends,\
//					h_graph_mask,\
//					h_updating_graph_mask,\
//					h_graph_visited,\
//					h_cost,\
//					is_updating_active_side\
//					);
//		}
//	}
//}
inline void scheduler(\
		const unsigned &start_row_index,\
		const unsigned &tile_step,
		unsigned *h_graph_heads,\
		unsigned *h_graph_ends,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_cost,\
		unsigned *tile_offsets,
		int *is_empty_tile,\
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
			if (is_empty_tile[tile_id] || !is_active_side[row_id]) {
				continue;
			}
			// Kernel
			unsigned bound_edge_i;
			if (NUM_TILES - 1 != tile_id) {
				bound_edge_i = tile_offsets[tile_id + 1];
			} else {
				bound_edge_i = NEDGES;
			}
			bfs_kernel(\
					tile_offsets[tile_id],\
					bound_edge_i,\
					h_graph_heads,\
					h_graph_ends,\
					h_graph_mask,\
					h_updating_graph_mask,\
					h_graph_visited,\
					h_cost,\
					is_updating_active_side\
					);
		}

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
		int *is_empty_tile,\
		int *is_active_side,\
		int *is_updating_active_side)
{

	//printf("Start traversing the tree\n");
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;
		unsigned side_id;
		for (side_id = 0; side_id + ROW_STEP <= SIDE_LENGTH; side_id += ROW_STEP) {
			scheduler(\
					//side_id * SIDE_LENGTH,\
					//(side_id + ROW_STEP) * SIDE_LENGTH,
					side_id,
					ROW_STEP,
					h_graph_heads,\
					h_graph_ends,\
					h_graph_mask,\
					h_updating_graph_mask,\
					h_graph_visited,\
					h_cost,\
					tile_offsets,
					is_empty_tile,\
					is_active_side,
					is_updating_active_side);
		}
		scheduler(\
				//side_id * SIDE_LENGTH,\
				//NUM_TILES,
				side_id,
				SIDE_LENGTH - side_id,
				h_graph_heads,\
				h_graph_ends,\
				h_graph_mask,\
				h_updating_graph_mask,\
				h_graph_visited,\
				h_cost,\
				tile_offsets,
				is_empty_tile,\
				is_active_side,
				is_updating_active_side);

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
		for (unsigned side_id = 0; side_id < SIDE_LENGTH; ++side_id) {
			if (!is_updating_active_side[side_id]) {
				is_active_side[side_id] = 0;
				continue;
			}
			is_updating_active_side[side_id] = 0;
			is_active_side[side_id] = 1;
			stop = false;
			for (unsigned i = 0; i < TILE_WIDTH; ++i) {
				unsigned vertex_id = i + side_id * TILE_WIDTH;
				if (vertex_id == NNODES) {
					break;
				}
				if (1 == h_updating_graph_mask[vertex_id]) {
					h_updating_graph_mask[vertex_id] = 0;
					h_graph_mask[vertex_id] = 1;
					h_graph_visited[vertex_id] = 1;
				} else {
					h_graph_mask[vertex_id] = 0;
				}
			}
		}
	}
	while(!stop);
	double end_time = omp_get_wtime();
	printf("%d %lf\n", NUM_THREADS, (end_time - start_time));
}
///////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph
///////////////////////////////////////////////////////////////////////////////
void input( int argc, char** argv) 
{
	char *input_f;
	ROW_STEP = 16;
	//ROW_STEP = 2;
	
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
	//string prefix = string(input_f) + "_coo-tiled-" + to_string(TILE_WIDTH);
	string prefix = string(input_f) + "_col-16-coo-tiled-" + to_string(TILE_WIDTH);
	//string prefix = string(input_f) + "_col-2-coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
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
	unsigned *tile_offsets = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		fscanf(fin, "%u", tile_offsets + i);
	}
	fclose(fin);
	unsigned *h_graph_starts = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
	unsigned *h_graph_ends = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
	int *is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
	memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);
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
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
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
	//for (unsigned ROW_STEP = 1; ROW_STEP < 10000; ROW_STEP *= 2) {
	//printf("ROW_STEP: %u\n", ROW_STEP);
	//unsigned ROW_STEP = 16;
	//ROW_STEP = 16;
	//ROW_STEP = 2;//test
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
		memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
		is_active_side[0] = 1;
		memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);

		BFS(\
			h_graph_starts,\
			h_graph_ends,\
			h_graph_mask,\
			h_updating_graph_mask,\
			h_graph_visited,\
			h_cost,\
			tile_offsets,
			is_empty_tile,\
			is_active_side,\
			is_updating_active_side);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
#ifdef ONEDEBUG
		printf("Thread %u finished.\n", NUM_THREADS);
#endif
	}
	//}
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
	free( is_empty_tile);
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

