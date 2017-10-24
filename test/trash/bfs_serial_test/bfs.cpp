#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>

using std::string;
using std::to_string;

int	NUM_THREADS;
unsigned TILE_WIDTH;

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
		unsigned num_of_nodes,\
		int edge_list_size,\
		int *is_empty_tile,\
		int *is_active_side,\
		int *is_updating_active_side,\
		unsigned side_length,\
		unsigned num_tiles,\
		unsigned row_step\
		);

///////////////////////////////////////////////////////////////////////////////
// Main Program
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	start = omp_get_wtime();
	input( argc, argv);
}

///////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph
///////////////////////////////////////////////////////////////////////////////
void input( int argc, char** argv) 
{
	unsigned int num_of_nodes = 0;
	int edge_list_size = 0;
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
	fscanf(fin, "%u %u", &num_of_nodes, &edge_list_size);
	fclose(fin);
	unsigned side_length;
	if (num_of_nodes % TILE_WIDTH) {
		side_length = num_of_nodes / TILE_WIDTH + 1;
	} else {
		side_length = num_of_nodes / TILE_WIDTH;
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
	unsigned *h_graph_starts = (unsigned *) malloc(sizeof(unsigned) * edge_list_size);
	unsigned *h_graph_ends = (unsigned *) malloc(sizeof(unsigned) * edge_list_size);
	int *is_empty_tile = (int *) malloc(sizeof(int) * num_tiles);
	memset(is_empty_tile, 0, sizeof(int) * num_tiles);
	NUM_THREADS = 64;
	unsigned edge_bound = edge_list_size / NUM_THREADS;
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
		fscanf(fin, "%u %u", &num_of_nodes, &edge_list_size);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = edge_list_size;
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

	int *h_graph_mask = (int*) malloc(sizeof(int)*num_of_nodes);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*num_of_nodes);
	int *h_graph_visited = (int*) malloc(sizeof(int)*num_of_nodes);
	int* h_cost = (int*) malloc(sizeof(int)*num_of_nodes);
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
	//for (unsigned row_step = 1; row_step < 10000; row_step *= 2) {
	//printf("row_step: %u\n", row_step);
	unsigned row_step = 16;
	for (unsigned i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#ifndef ONEDEBUG
		//sleep(10);
#endif
		// Re-initializing
		memset(h_graph_mask, 0, sizeof(int)*num_of_nodes);
		h_graph_mask[source] = 1;
		memset(h_updating_graph_mask, 0, sizeof(int)*num_of_nodes);
		memset(h_graph_visited, 0, sizeof(int)*num_of_nodes);
		h_graph_visited[source] = 1;
		for (unsigned i = 0; i < num_of_nodes; ++i) {
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
			num_of_nodes,\
			edge_list_size,\
			is_empty_tile,\
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
	//}
	fclose(time_out);

	//Store the result into a file

#ifdef ONEDEBUG
	NUM_THREADS = 64;
	omp_set_num_threads(NUM_THREADS);
	unsigned num_lines = num_of_nodes / NUM_THREADS;
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
		bound_index = num_of_nodes;
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

inline void scheduler(\
		const unsigned &start_row_index,\
		const unsigned &bound_row_index,\
		unsigned *h_graph_heads,\
		unsigned *h_graph_ends,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_cost,\
		unsigned *tile_offsets,
		const int &edge_list_size,\
		const unsigned &side_length,\
		const unsigned &num_tiles,\
		int *is_empty_tile,\
		int *is_active_side,
		int *is_updating_active_side\
		)
{
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned col_id = 0; col_id < side_length; ++col_id) {
		for (unsigned row_id = start_row_index; row_id < bound_row_index; ++row_id) {
			unsigned tile_id = row_id * side_length + col_id;
			//if (is_empty_tile[tile_id] ) {
			//	continue;
			//}
			if (is_empty_tile[tile_id] || !is_active_side[row_id]) {
				continue;
			}
			//bfs_kernel();
			unsigned bound_edge_i;
			if (num_tiles - 1 != tile_id) {
				bound_edge_i = tile_offsets[tile_id + 1];
			} else {
				bound_edge_i = edge_list_size;
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
			//for (unsigned edge_i = tile_offsets[tile_id]; edge_i < bound_edge_i; ) {
			//	unsigned head = h_graph_heads[edge_i];
			//	if (0 == h_graph_mask[head]) {
			//		++edge_i;
			//		continue;
			//	}
			//	while (h_graph_heads[edge_i] == head) {
			//		unsigned end = h_graph_ends[edge_i];
			//		if (!h_graph_visited[end]) {
			//			h_cost[end] = h_cost[head] + 1;
			//			h_updating_graph_mask[end] = 1;
			//			is_updating_active_side[end/TILE_WIDTH] = 1;
			//		}
			//		++edge_i;
			//	}
			//}
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
		unsigned num_of_nodes,\
		int edge_list_size,\
		int *is_empty_tile,\
		int *is_active_side,\
		int *is_updating_active_side,\
		unsigned side_length,\
		unsigned num_tiles,\
		unsigned row_step
		)
{

	//printf("Start traversing the tree\n");
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;

		for (unsigned side_id = 0; side_id < side_length; ) {
			//if (!is_active_side[side_id]) {
			//	++side_id;
			//	continue;
			//}
			if (side_id + row_step < side_length) {
				scheduler(\
						side_id,\
						side_id + row_step,\
						h_graph_heads,\
						h_graph_ends,\
						h_graph_mask,\
						h_updating_graph_mask,\
						h_graph_visited,\
						h_cost,\
						tile_offsets,
						edge_list_size,\
						side_length,\
						num_tiles,\
						is_empty_tile,\
						is_active_side,
						is_updating_active_side
						);
				side_id += row_step;
			} else {
				scheduler(\
						side_id,\
						side_length,\
						h_graph_heads,\
						h_graph_ends,\
						h_graph_mask,\
						h_updating_graph_mask,\
						h_graph_visited,\
						h_cost,\
						tile_offsets,
						edge_list_size,\
						side_length,\
						num_tiles,\
						is_empty_tile,\
						is_active_side,
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
		//			bound_edge_i = edge_list_size;
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
		//for(unsigned int nid=0; nid< num_of_nodes ; nid++ )
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
			for (unsigned i = 0; i < TILE_WIDTH; ++i) {
				unsigned vertex_id = i + side_id * TILE_WIDTH;
				if (vertex_id == num_of_nodes) {
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
