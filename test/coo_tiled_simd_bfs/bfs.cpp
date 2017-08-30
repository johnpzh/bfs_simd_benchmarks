#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <map>
#include <immintrin.h>
//#define NUM_THREAD 4

using std::string;
using std::to_string;
using std::unordered_map;
using std::map;
using std::pair;

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

//Structure to hold a node information
//struct Node
//{
//	//int starting;
//	//int num_of_edges;
//	unsigned start;
//	unsigned outdegree;
//};
//struct Tile
//{
//	unsigned num_vertices;
//	unsigned num_edges;
//	unsigned vertices_offset;
//	unsigned edges_offset;
//	//unsigned *vertices;
//	//unsigned *starts;
//	//unsigned *edges;
//	//unsigned vertices[2048];
//	//unsigned starts[2048];
//	//unsigned edges[7000];
//
//	void init(unsigned n_vertices, unsigned n_edges, unsigned v_offset, unsigned e_offset) {
//		num_vertices = n_vertices;
//		num_edges = n_edges;
//		vertices_offset = v_offset;
//		edges_offset = e_offset;
//		//vertices = (unsigned *) malloc(sizeof(unsigned) * num_vertices);
//		//starts = (unsigned *) malloc(sizeof(unsigned) * num_vertices);
//		//edges = (unsigned *) malloc(sizeof(unsigned) * num_edges);
//	}
//
//	//~Tile() {
//	//	free(vertices);
//	//	free(starts);
//	//	free(edges);
//	//}
//};

//typedef unordered_map<unsigned, unsigned[2]> hashmap_csr;
//typedef unordered_map<unsigned, Node> hashmap_csr;

int	NUM_THREADS;
unsigned TILE_WIDTH;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


void BFSGraph(int argc, char** argv);
void BFS_kernel(\
		//int *h_graph_indices,
		//int *h_graph_heads,
		//int *h_graph_data,
		//Tile *h_graph_tiles,
		unsigned *h_graph_heads,\
		unsigned *h_graph_ends,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		//int *h_graph_edges,
		int *h_cost,\
		unsigned *tile_offsets,
		//unsigned *indices_offsets,
		unsigned num_of_nodes,\
		int edge_list_size,\
		//unsigned num_of_indices,
		int *not_empty_tile,\
		int *is_active_side,\
		int *is_updating_active_side,\
		unsigned side_length,\
		unsigned num_tiles\
		);

///////////////////////////////////////////////////////////////////////////////
// Main Program
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	start = omp_get_wtime();
	BFSGraph( argc, argv);
}

///////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph
///////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
	unsigned int num_of_nodes = 0;
	int edge_list_size = 0;
	int num_of_indices;
	char *input_f;
	
	if(argc < 3){
		input_f = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
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
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	//fscanf(fin, "%u %u %u", &num_of_nodes, &edge_list_size, &num_of_indices);
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
	//fname = prefix + "-indices_offsets";
	//fin = fopen(fname.c_str(), "r");
	//if (!fin) {
	//	fprintf(stderr, "cannot open file: %s\n", fname.c_str());
	//	exit(1);
	//}
	//unsigned *indices_offsets = (unsigned *) malloc(num_tiles * sizeof(unsigned));
	//for (unsigned i = 0; i < num_tiles; ++i) {
	//	fscanf(fin, "%u", indices_offsets + i);
	//}
	//fclose(fin);
	//// Read file Offsets
	//fname = prefix + "-file_offsets";
	//fin = fopen(fname.c_str(), "r");
	//if (!fin) {
	//	fprintf(stderr, "cannot open file: %s\n", fname.c_str());
	//	exit(1);
	//}
	//NUM_THREADS = 64;
	//unsigned *file_offsets = (unsigned *) malloc(NUM_THREADS * sizeof(unsigned));
	//for (unsigned i = 0; i < NUM_THREADS; ++i) {
	//	fscanf(fin, "%u", file_offsets + i);
	//}
	//fclose(fin);

	//hashmap_csr *tiles_indices = new hashmap_csr[num_tiles];
	
	//int *h_graph_edges = (int *) malloc(sizeof(int) * edge_list_size);
	//int *h_graph_indices = (int *) malloc(sizeof(int) * num_of_indices);
	//int *h_graph_heads = (int *) malloc(sizeof(int) * num_of_indices);
	//int *h_graph_outdegrees = (int *)  = tid * malloc(sizeof(int) * num_of_indices);
	//unsigned *n1s = (unsigned *) malloc(edge_list_size * sizeof(unsigned));
	//unsigned *n2s = (unsigned *) malloc(edge_list_size * sizeof(unsigned));
	//int *h_graph_data = (int *) malloc(sizeof(int) * (2*num_of_indices + edge_list_size));
	//Tile *h_graph_tiles = new Tile[num_tiles];
	unsigned *h_graph_heads = (unsigned *) _mm_malloc(sizeof(unsigned) * edge_list_size, ALIGNED_BYTES);
	unsigned *h_graph_ends = (unsigned *) malloc(sizeof(unsigned) * edge_list_size);
	//int *is_empty_tile = (int *) malloc(sizeof(int) * num_tiles);
	int *not_empty_tile = (int *) malloc(sizeof(int) * num_tiles);
	memset(not_empty_tile, 0, sizeof(int) * num_tiles);
	NUM_THREADS = 64;
	unsigned edge_bound = edge_list_size / NUM_THREADS;
	//unsigned bound_tiles = num_tiles/NUM_THREADS;// number of tiles per file
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
		//fscanf(fin, "%u %u %u", &num_of_nodes, &edge_list_size, &num_of_indices);
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
		h_graph_heads[index] = n1;
		h_graph_ends[index] = n2;
		unsigned n1_id = n1 / TILE_WIDTH;
		unsigned n2_id = n2 / TILE_WIDTH;
		unsigned tile_id = n1_id * side_length + n2_id;
		not_empty_tile[tile_id] = 1;
	}

	//unsigned offset_file = tid * bound_tiles;
	//unsigned bound_tile_id;
	//if (NUM_THREADS - 1 != tid) {
	//	bound_tile_id = offset_file + bound_tiles;
	//} else {
	//	bound_tile_id = num_tiles;
	//}
	//for (unsigned tile_id = offset_file; tile_id < bound_tile_id; ++tile_id) {
	//	unsigned num_indices;
	//	unsigned num_edges;
	//	// Read number of indices, number of edges
	//	fscanf(fin, "%u %u", &num_indices, &num_edges);
	//	if (0 == num_indices) {
	//		is_empty_tile[tile_id] = 1;
	//		continue;
	//	}
	//	unsigned vertices_offset = 2 * indices_offsets[tile_id] + tile_offsets[tile_id];
	//	//unsigned starts_offset = vertices_offset + num_indices;
	//	unsigned edges_offset = vertices_offset + 2*num_indices;
	//	Tile &tile_ref = h_graph_tiles[tile_id];
	//	tile_ref.init(num_indices, num_edges, vertices_offset, edges_offset);
	//	// Read indices
	//	for (unsigned i = 0; i < num_indices; ++i) {
	//		unsigned index;
	//		unsigned start;
	//		unsigned outdegree;
	//		//unsigned index_i = i + indices_offsets[tile_id];
	//		fscanf(fin, "%u %u %u", &index, &start, &outdegree);
	//		index--;
	//		start += tile_ref.edges_offset;
	//		//start += tile_offsets[tile_id];
	//		//tiles_indices[tile_id][index][0] = start;
	//		//tiles_indices[tile_id][index][1] = outdegree;
	//		//h_graph_indices[index_i] = index;
	//		//h_graph_heads[index_i] = start;
	//		//h_graph_outdegrees[index_i] = outdegree;
	//		//tile_ref.vertices[i] = index;
	//		//tile_ref.starts[i] = start;
	//		h_graph_data[tile_ref.vertices_offset + i] = index;
	//		h_graph_data[tile_ref.vertices_offset + num_indices + i] = start;
	//	}
	//	// Read edges
	//	for (unsigned i = 0; i < num_edges; ++i) {
	//		//unsigned index = i + tile_offsets[tile_id];
	//		int end;
	//		fscanf(fin, "%d", &end);
	//		end--;
	//		//h_graph_edges[index] = end;
	//		//tile_ref.edges[i] = end;
	//		h_graph_data[tile_ref.edges_offset + i] = end;
	//	}
	//}
}
	/////////////////////////////////////////
	//Empty Tiles Percentage
	unsigned empty_count = 0;
	for (unsigned i = 0; i < num_tiles; ++i) {
		if (!not_empty_tile[i]) {
			++empty_count;
		}
	}
	printf("empty tile percentage: %lf\n", empty_count * 1.0 / num_tiles);
	//exit(3);
	///////////////////////////////////////////////
	
	// Read nneibor
	//fname = prefix + "-nneibor";
	//fin = fopen(fname.c_str(), "r");
	//if (!fin) {
	//	fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
	//	exit(1);
	//}
	//unsigned *nneibor = (unsigned *) malloc(num_of_nodes * sizeof(unsigned));
	//for (unsigned i = 0; i < num_of_nodes; ++i) {
	//	fscanf(fin, "%u", nneibor + i);
	//}
	// End Input real dataset
	/////////////////////////////////////////////////////////////////////

	//unsigned NUM_CORE = 64;
	//omp_set_num_threads(NUM_CORE);
	//string file_prefix = input_f;
	//string file_name = file_prefix + "-v0.txt";
	//FILE *finput = fopen(file_name.c_str(), "r");
	//fscanf(finput, "%u", &num_of_nodes);
	//fclose(finput);
	//unsigned num_lines = num_of_nodes / NUM_CORE;
	//Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*num_of_nodes);
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
	unsigned run_count = 1;
#else
	unsigned run_count = 9;
#endif
	// BFS
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#ifndef ONEDEBUG
		sleep(10);
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

		BFS_kernel(\
			//int *h_graph_indices,
			//int *h_graph_heads,
			//h_graph_data,
			//h_graph_tiles,
			h_graph_heads,\
			h_graph_ends,\
			h_graph_mask,\
			h_updating_graph_mask,\
			h_graph_visited,\
			//h_graph_edges,
			h_cost,\
			tile_offsets,
			//unsigned *indices_offsets,
			num_of_nodes,\
			edge_list_size,\
			//unsigned num_of_indices,
			not_empty_tile,\
			is_active_side,\
			is_updating_active_side,\
			side_length,\
			num_tiles\
			);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
#ifdef ONEDEBUG
		printf("Thread %u finished.\n", NUM_THREADS);
#endif
	}
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

	//if (tid != NUM_THREADS - 1) {
	//	for (unsigned i = 0; i < num_lines; ++i) {
	//		unsigned index = i + offset;
	//		fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
	//	}
	//} else {
	//	for (unsigned i = 0; i + offset < num_of_nodes; ++i) {
	//		unsigned index = i + offset;
	//		fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
	//	}
	//}
	fclose(fpo);
}
#endif
	//printf("Result stored in result.txt\n");

	// cleanup memory
	//free(nneibor);
	//free(n1s);
	//free(n2s);
	//free( h_graph_nodes);
	//free( h_graph_indices);
	//free( h_graph_heads);
	//free( h_graph_outdegrees);
	//free( h_graph_data);
	//delete [] h_graph_tiles;
	//free( h_graph_edges);
	_mm_free( h_graph_heads);
	free( h_graph_ends);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	free( tile_offsets);
	//free( indices_offsets);
	//delete [] tiles_indices;
	free( not_empty_tile);
	free( is_active_side);
	free( is_updating_active_side);
	//free( file_offsets);
}

//void BFS_kernel(\
//		Node *h_graph_nodes,\
//		int *h_graph_mask,\
//		int *h_updating_graph_mask,\
//		int *h_graph_visited,\
//		int *h_graph_edges,\
//		int *h_cost,\
//		unsigned *offsets,\
//		unsigned num_of_nodes,\
//		int edge_list_size\
//		)
void BFS_kernel(\
		//int *h_graph_indices,
		//int *h_graph_heads,
		//int *h_graph_data,
		//Tile *h_graph_tiles,
		unsigned *h_graph_heads,\
		unsigned *h_graph_ends,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		//int *h_graph_edges,
		int *h_cost,\
		unsigned *tile_offsets,
		//unsigned *indices_offsets,
		unsigned num_of_nodes,\
		int edge_list_size,\
		//unsigned num_of_indices,
		int *not_empty_tile,\
		int *is_active_side,\
		int *is_updating_active_side,\
		unsigned side_length,\
		unsigned num_tiles\
		)
{
	__m512i one_v = _mm512_set1_epi32(1);
	__m512i zero_v = _mm512_set1_epi32(0);
	__m512i minusone_v = _mm512_set1_epi32(-1);

#ifdef ONDEBUG
#endif

	//printf("Start traversing the tree\n");
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;

#pragma omp parallel for schedule(dynamic, 1)
		//for(unsigned int nid = 0; nid < num_of_nodes; nid++ )
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
		for (unsigned side_id = 0; side_id < side_length; ++side_id) {
			if (!is_active_side[side_id]) {
				continue;
			}
			is_active_side[side_id] = 0;
			unsigned start_tile_id = side_id * side_length;
			unsigned bound_tile_id = start_tile_id + side_length;
			for (unsigned tile_id = start_tile_id; \
					tile_id < bound_tile_id;\
					++tile_id) {
				if (!not_empty_tile[tile_id]) {
					continue;
				}
				unsigned bound_edge_i;
				if (num_tiles - 1 != tile_id) {
					bound_edge_i = tile_offsets[tile_id + 1];
				} else {
					bound_edge_i = edge_list_size;
				}
				//for (unsigned edge_i = tile_offsets[tile_id]; \
				//		edge_i < bound_edge_i; \
				//		) {
				//	unsigned head = h_graph_heads[edge_i];
				//	if (0 == h_graph_mask[head]) {
				//		edge_i++;
				//		continue;
				//	}
				//	while (h_graph_heads[edge_i] == head) {
				//		unsigned end = h_graph_ends[edge_i];
				//		if (!h_graph_visited[end]) {
				//			h_cost[end] = h_cost[head] + 1;
				//			h_updating_graph_mask[end] = 1;
				//			is_updating_active_side[end/TILE_WIDTH] = 1;
				//		}
				//		edge_i++;
				//	}
				//}
				unsigned edge_i;
				for (edge_i = tile_offsets[tile_id]; \
						edge_i <= bound_edge_i - NUM_P_INT; \
						edge_i += NUM_P_INT) {
					__m512i head_v = _mm512_loadu_si512(h_graph_heads + edge_i);
					__m512i active_flag_v = _mm512_i32gather_epi32(head_v, h_graph_mask, sizeof(int));
					__mmask16 is_active_m = _mm512_cmpeq_epi32_mask(active_flag_v, one_v);
					if (*((short *)(&is_active_m)) == 0) {
						continue;
					}
					__m512i end_v = _mm512_loadu_si512(h_graph_ends + edge_i);
					__m512i visited_flag_v = _mm512_mask_i32gather_epi32(one_v, is_active_m, end_v, h_graph_visited, sizeof(int));
					__mmask16 not_visited_m = _mm512_cmpeq_epi32_mask(visited_flag_v, zero_v);
					if (*((short *)(&not_visited_m)) == 0) {
						continue;
					}
					_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, end_v, one_v, sizeof(int));
					__m512i cost_head_v = _mm512_i32gather_epi32(head_v, h_cost, sizeof(int));
					__m512i cost_end_v = _mm512_add_epi32(cost_head_v, one_v);
					_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, end_v, cost_end_v, sizeof(int));
					__m512i TILE_WIDTH_V = _mm512_set1_epi32(TILE_WIDTH);
					__m512i side_i_v = _mm512_div_epi32(end_v, TILE_WIDTH_V);
					_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_i_v, one_v, sizeof(int));
				}
				__m512i edge_i_v = _mm512_set_epi32(edge_i + 15, edge_i + 14, edge_i + 13, edge_i + 12,\
													edge_i + 11, edge_i + 10, edge_i + 9, edge_i + 8,\
													edge_i + 7, edge_i + 6, edge_i + 5, edge_i + 4,\
													edge_i + 3, edge_i + 2, edge_i + 1, edge_i);
				__m512i bound_edge_i_v = _mm512_set1_epi32(bound_edge_i);
				__mmask16 in_range_m = _mm512_cmplt_epi32_mask(edge_i_v, bound_edge_i_v);
				//__m512i head_v = _mm512_loadu_si512(h_graph_heads + edge_i);
				__m512i head_v = _mm512_mask_loadu_epi32(minusone_v, in_range_m, h_graph_heads + edge_i);
				__m512i active_flag_v = _mm512_mask_i32gather_epi32(zero_v, in_range_m, head_v, h_graph_mask, sizeof(int));
				__mmask16 is_active_m = _mm512_cmpeq_epi32_mask(active_flag_v, one_v);
				if (*((short *)(&is_active_m)) == 0) {
					continue;
				}
				//__m512i end_v = _mm512_loadu_si512(h_graph_ends + edge_i);
				__m512i end_v = _mm512_mask_loadu_epi32(minusone_v, in_range_m, h_graph_ends + edge_i);
				__m512i visited_flag_v = _mm512_mask_i32gather_epi32(one_v, is_active_m, end_v, h_graph_visited, sizeof(int));
				__mmask16 not_visited_m = _mm512_cmpeq_epi32_mask(visited_flag_v, zero_v);
				if (*((short *)(&not_visited_m)) == 0) {
					continue;
				}
				_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, end_v, one_v, sizeof(int));
				__m512i cost_head_v = _mm512_mask_i32gather_epi32(zero_v, not_visited_m, head_v, h_cost, sizeof(int));
				__m512i cost_end_v = _mm512_add_epi32(cost_head_v, one_v);
				_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, end_v, cost_end_v, sizeof(int));
				__m512i TILE_WIDTH_V = _mm512_set1_epi32(TILE_WIDTH);
				__m512i side_i_v = _mm512_div_epi32(end_v, TILE_WIDTH_V);
				_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_i_v, one_v, sizeof(int));
			}
		}
#pragma omp parallel for schedule(dynamic, 1)
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
				continue;
			}
			is_updating_active_side[side_id] = 0;
			is_active_side[side_id] = 1;
			stop = false;
			unsigned bound_vertex_id;
			if (side_length - 1 != side_id) {
				bound_vertex_id = side_id * TILE_WIDTH + TILE_WIDTH;
			} else {
				bound_vertex_id = num_of_nodes;
			}
			//for (unsigned vertex_id = side_id * TILE_WIDTH; \
			//		vertex_id < bound_vertex_id; \
			//		++vertex_id) {
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
					vertex_id <= bound_vertex_id - NUM_P_INT; \
					vertex_id += NUM_P_INT) {
				__m512i updating_flag_v = _mm512_loadu_si512(h_updating_graph_mask + vertex_id);
				__mmask16 is_updating_m = _mm512_cmpeq_epi32_mask(updating_flag_v, one_v);
				if (*((short *)(&is_updating_m)) == 0) {
					continue;
				}
				_mm512_mask_storeu_epi32(h_updating_graph_mask + vertex_id, is_updating_m, zero_v);
				_mm512_storeu_si512(h_graph_mask + vertex_id, updating_flag_v);
				_mm512_mask_storeu_epi32(h_graph_visited + vertex_id, is_updating_m, one_v);
			}
			__m512i vertex_id_v = _mm512_set_epi32(vertex_id + 15, vertex_id + 14, vertex_id + 13, vertex_id + 12,\
													vertex_id + 11, vertex_id + 10, vertex_id + 9, vertex_id + 8,\
													vertex_id + 7, vertex_id + 6, vertex_id + 5, vertex_id + 4,\
													vertex_id + 3, vertex_id + 2, vertex_id + 1, vertex_id);
			__m512i bound_vertex_id_v = _mm512_set1_epi32(bound_vertex_id);
			__mmask16 in_range_m = _mm512_cmplt_epi32_mask(vertex_id_v, bound_vertex_id_v);
			__m512i updating_flag_v = _mm512_mask_loadu_epi32(zero_v, in_range_m, h_updating_graph_mask + vertex_id);
			__mmask16 is_updating_m = _mm512_cmpeq_epi32_mask(updating_flag_v, one_v);
			if (*((short *)(&is_updating_m)) == 0) {
				continue;
			}
			_mm512_mask_storeu_epi32(h_updating_graph_mask + vertex_id, is_updating_m, zero_v);
			_mm512_mask_storeu_epi32(h_graph_mask + vertex_id, in_range_m, updating_flag_v);
			_mm512_mask_storeu_epi32(h_graph_visited + vertex_id, is_updating_m, one_v);

		}
	}
	while(!stop);
	double end_time = omp_get_wtime();
	printf("%d %lf\n", NUM_THREADS, (end_time - start_time));
}
