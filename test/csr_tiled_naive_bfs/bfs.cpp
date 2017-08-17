#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
//#define NUM_THREAD 4

using std::string;
using std::to_string;
using std::unordered_map;

typedef unordered_map<unsigned, unsigned[2]> hashmap_csr;

int	NUM_THREADS;
unsigned TILE_WIDTH;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

//Structure to hold a node information
struct Node
{
	int starting;
	int num_of_edges;
};

void BFSGraph(int argc, char** argv);
void BFS_kernel(\
		hashmap_csr *tiles_indices,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_graph_edges,\
		int *h_cost,\
		unsigned *tile_offsets,\
		unsigned num_of_nodes,\
		int edge_list_size,\
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
	string prefix = string(input_f) + "_csr-tiled-" + to_string(TILE_WIDTH);
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
	fname = prefix + "-tile_offsets";
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

	hashmap_csr *tiles_indices = (hashmap_csr *) malloc(sizeof(hashmap_csr) * num_tiles);
	
	int *h_graph_edges = (int *) malloc(sizeof(int) * edge_list_size);
	//unsigned *n1s = (unsigned *) malloc(edge_list_size * sizeof(unsigned));
	//unsigned *n2s = (unsigned *) malloc(edge_list_size * sizeof(unsigned));
	int *is_empty_tile = (int *) malloc(sizeof(int) * num_tiles);
	memset(is_empty_tile, 0, sizeof(int) * num_tiles);
	NUM_THREADS = 64;
	//unsigned edge_bound = edge_list_size / NUM_THREADS;
	unsigned bound_tiles = num_tiles/NUM_THREADS;// number of tiles per file
#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
{
	unsigned tid = omp_get_thread_num();
	//unsigned offset = tid * edge_bound;
	//unsigned file_offset = file_offsets[tid];
	fname = prefix + "-" + to_string(tid);
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
		exit(1);
	}
	if (0 == tid) {
		fscanf(fin, "%u %u", &num_of_nodes, &edge_list_size);
	}
	unsigned offset_file = tid * bound_tiles;
	if (NUM_THREADS - 1 != tid) {
		for (unsigned i = 0; i < bound_tiles; ++i) {
			unsigned tile_id = i + offset_file;
			unsigned num_indices;
			unsigned num_edges;
			fscanf(fin, "%u %u", &num_indices, &num_edges);
			if (0 == num_indices) {
				is_empty_tile[tile_id] = 1;
				continue;
			}
			for (unsigned i_indices = 0; i_indices < num_indices; ++i_indices) {
				unsigned index;
				unsigned start;
				unsigned outdegree;
				fscanf(fin, "%u %u %u", &index, &start, &outdegree);
				index--;
				start += tile_offsets[tile_id];
				//tiles_indices[tile_id][index] = {start, outdegree};
				tiles_indices[tile_id][index][0] = start;
				tiles_indices[tile_id][index][1] = outdegree;
			}
			for (unsigned i_edges = 0; i_edges < num_edges; ++i_edges) {
				unsigned index = i_edges + tile_offsets[tile_id];
				fscanf(fin, "%d", h_graph_edges + index);
			}
		}
	} else { // the last file contains maybe more tiles
		for (unsigned i = 0; i + offset_file < num_tiles; ++i) {
			unsigned tile_id = i + offset_file;
			unsigned num_indices;
			unsigned num_edges;
			fscanf(fin, "%u %u", &num_indices, &num_edges);
			if (0 == num_indices) {
				is_empty_tile[tile_id] = 1;
				continue;
			}
			for (unsigned i_indices = 0; i_indices < num_indices; ++i_indices) {
				unsigned index;
				unsigned start;
				unsigned outdegree;
				fscanf(fin, "%u %u %u", &index, &start, &outdegree);
				index--;
				start += tile_offsets[tile_id];
				//tiles_indices[tile_id][index] = {start, outdegree};
				tiles_indices[tile_id][index][0] = start;
				tiles_indices[tile_id][index][1] = outdegree;
			}
			for (unsigned i_edges = 0; i_edges < num_edges; ++i_edges) {
				unsigned index = i_edges + tile_offsets[tile_id];
				fscanf(fin, "%d", h_graph_edges + index);
			}
		}
	}


	//if (NUM_THREADS - 1 != tid) {
	//	for (unsigned i = 0; i < edge_bound; ++i) {
	//		unsigned index = i + offset;
	//		unsigned n1;
	//		unsigned n2;
	//		fscanf(fin, "%u %u", &n1, &n2);
	//		n1--;
	//		n2--;
	//		n1s[index] = n1;
	//		n2s[index] = n2;
	//	}
	//} else {
	//	for (unsigned i = 0; i + offset < edge_list_size; ++i) {
	//		unsigned index = i + offset;
	//		unsigned n1;
	//		unsigned n2;
	//		fscanf(fin, "%u %u", &n1, &n2);
	//		n1--;
	//		n2--;
	//		n1s[index] = n1;
	//		n2s[index] = n2;
	//	}
	//}
	//fclose(fin);
}
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
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*num_of_nodes);
	int *h_graph_mask = (int*) malloc(sizeof(int)*num_of_nodes);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*num_of_nodes);
	int *h_graph_visited = (int*) malloc(sizeof(int)*num_of_nodes);
	int* h_cost = (int*) malloc(sizeof(int)*num_of_nodes);
//#pragma omp parallel private(file_name, finput)
//{
//	unsigned thid = omp_get_thread_num();
//	file_name = file_prefix + "-v" + to_string(thid) + ".txt";
//	finput = fopen(file_name.c_str(), "r");
//	if (0 == thid) {
//		fscanf(finput, "%u", &num_of_nodes);
//	}
//	unsigned start, edgeno;
//	unsigned offset = thid * num_lines; // here I assume the num_of_nodes can be divided by NUM_CORE
//	for (unsigned i = 0; i < num_lines; ++i) {
//		fscanf(finput, "%u %u", &start, &edgeno);
//		unsigned index = i + offset;
//		h_graph_nodes[index].starting = start;
//		h_graph_nodes[index].num_of_edges = edgeno;
//		h_graph_mask[index] = 0;
//		h_updating_graph_mask[index] = 0;
//		h_graph_visited[index] = 0;
//		h_cost[index] = -1;
//	}
//	fclose(finput);
//}
	//unsigned edge_start = 0;
	//for (unsigned i = 0; i < num_of_nodes; ++i) {
	//	h_graph_nodes[i].starting = edge_start;
	//	h_graph_nodes[i].num_of_edges = nneibor[i];
	//	edge_start += nneibor[i];
	//	//h_graph_mask[i] = 0;
	//	//h_updating_graph_mask[i] = 0;
	//	//h_graph_visited[i] = 0;
	//	//h_cost[i] = -1;
	//}

	unsigned source = 0;
	//file_name = file_prefix + "-e0.txt";
	//finput = fopen(file_name.c_str(), "r");
	//fscanf(finput, "%u %u", &source, &edge_list_size);
	//fclose(finput);
	//h_graph_mask[source] = 1;
	//h_graph_visited[source] = 1;
	//h_cost[source] = 0;
	//num_lines = edge_list_size / NUM_CORE;
	//int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
//#pragma omp parallel private(file_name, finput)
//{
//	unsigned thid = omp_get_thread_num();
//	file_name = file_prefix + "-e" + to_string(thid) + ".txt";
//	finput = fopen(file_name.c_str(), "r");
//	if (0 == thid) {
//		fscanf(finput, "%u %u", &source, &edge_list_size);
//	}
//	unsigned id;
//	unsigned cost;
//	unsigned offset = thid * num_lines;
//	if (NUM_CORE - 1 != thid) {
//		for (unsigned i = 0; i < num_lines; ++i) {
//			fscanf(finput, "%u %u", &id, &cost);
//			unsigned index = i + offset;
//			h_graph_edges[index] = id;
//		}
//	} else {
//		for (unsigned i = 0; fscanf(finput, "%u %u", &id, &cost) != EOF; ++i) {
//			unsigned index = i + offset;
//			h_graph_edges[index] = id;
//		}
//	}
//	fclose(finput);
//}
//	NUM_THREADS = 256;
//#pragma omp parallel for num_threads(NUM_THREADS)
//	for (unsigned i = 0; i < edge_list_size; ++i) {
//		h_graph_edges[i] = n2s[i];
//	}

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("Input finished: %s\n", input_f);
#endif
	// BFS
	//for (unsigned i = 0; i < 9; ++i) {
	for (unsigned i = 0; i < 1; ++i) {
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

		BFS_kernel(\
				tiles_indices,\
				h_graph_mask,\
				h_updating_graph_mask,\
				h_graph_visited,\
				h_graph_edges,\
				h_cost,\
				tile_offsets,\
				num_of_nodes,\
				edge_list_size,\
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
	for (unsigned i = 0; i < num_lines; ++i) {
		unsigned index = i + offset;
		fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
	}
	fclose(fpo);
}
#endif
	//printf("Result stored in result.txt\n");

	// cleanup memory
	//free(nneibor);
	//free(n1s);
	//free(n2s);
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	free( tile_offsets);
	free( tiles_indices);
	free( is_empty_tile);
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
		hashmap_csr *tiles_indices,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_graph_edges,\
		int *h_cost,\
		unsigned *tile_offsets,\
		unsigned num_of_nodes,\
		int edge_list_size,\
		unsigned side_length,\
		unsigned num_tiles\
		)
{

	//printf("Start traversing the tree\n");
	unsigned test_count = 0;//test
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;

#pragma omp parallel for
		for(unsigned int nid = 0; nid < num_of_nodes; nid++ )
		{
			if (h_graph_mask[nid] == 1) {
				h_graph_mask[nid]=0;
				unsigned row_id = nid / TILE_WIDTH;
				unsigned lower_tile = row_id * side_length;
				unsigned upper_tile = lower_tile + side_length;
				unsigned lower_nid = offsets[lower_tile];
				unsigned upper_nid;
				if (upper_tile == num_tiles) {
					upper_nid = edge_list_size;
				} else {
					upper_nid = offsets[upper_tile];
				}
				for (unsigned i = lower_nid; i < upper_nid; ++i) {
					if (n1s[i] == nid) {
						unsigned n2 = n2s[i];
						if(!h_graph_visited[n2])
						{
							h_cost[n2]=h_cost[nid]+1;
							test_count++;//test
							h_updating_graph_mask[n2]=1;
						}
					}
				}
				//int next_starting = h_graph_nodes[nid].starting + h_graph_nodes[nid].num_of_edges;
				//for(int i = h_graph_nodes[nid].starting; \
				//		i < next_starting; \
				//		i++)
				//{
				//	int id = h_graph_edges[i];
				//	if(!h_graph_visited[id])
				//	{
				//		h_cost[id]=h_cost[nid]+1;
				//		h_updating_graph_mask[id]=1;
				//	}
				//}
			}
		}
#pragma omp parallel for
		for(unsigned int nid=0; nid< num_of_nodes ; nid++ )
		{
			if (h_updating_graph_mask[nid] == 1) {
				h_graph_mask[nid]=1;
				h_graph_visited[nid]=1;
				stop = false;
				h_updating_graph_mask[nid]=0;
			}
		}
		printf("test_count: %u\n", test_count);//test
	}
	while(!stop);
	double end_time = omp_get_wtime();
	printf("%d %lf\n", NUM_THREADS, (end_time - start_time));
}
