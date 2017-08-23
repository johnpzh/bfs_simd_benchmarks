#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>
//#define NUM_THREAD 4

using std::string;
using std::to_string;

int	NUM_THREADS;

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
		Node *h_graph_nodes,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_graph_edges,\
		int *h_cost,\
		unsigned num_of_nodes,\
		int edge_list_size\
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
	
	if(argc < 2){
	//Usage(argc, argv);
	//Usage( argv);
	//exit(0);
	//NUM_THREADS = 1;
	input_f = "/home/zpeng/benchmarks/data/pokec/untiled_bak/soc-pokec-relationships.txt";
	} else {
	//NUM_THREADS = atoi(argv[1]);
	input_f = argv[1];
	}

	/////////////////////////////////////////////////////////////////////
	// Input real dataset
	/////////////////////////////////////////////////////////////////////
	string prefix = string(input_f) + "_untiled";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	fscanf(fin, "%u %u", &num_of_nodes, &edge_list_size);
	fclose(fin);
	//memset(nneibor, 0, num_of_nodes * sizeof(unsigned));
	//for (unsigned i = 0; i < num_of_nodes; ++i) {
	//	grah.nneibor[i] = 0;
	//}
	unsigned *n1s = (unsigned *) malloc(edge_list_size * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(edge_list_size * sizeof(unsigned));
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
		fscanf(fin, "%u %u\n", &num_of_nodes, &edge_list_size);
	}
	if (NUM_THREADS - 1 != tid) {
		for (unsigned i = 0; i < edge_bound; ++i) {
			unsigned index = i + offset;
			unsigned n1;
			unsigned n2;
			fscanf(fin, "%u %u", &n1, &n2);
			n1--;
			n2--;
			n1s[index] = n1;
			n2s[index] = n2;
		}
	} else {
		for (unsigned i = 0; i + offset < edge_list_size; ++i) {
			unsigned index = i + offset;
			unsigned n1;
			unsigned n2;
			fscanf(fin, "%u %u", &n1, &n2);
			n1--;
			n2--;
			n1s[index] = n1;
			n2s[index] = n2;
		}
	}
	fclose(fin);
}
	// Read nneibor
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
		exit(1);
	}
	unsigned *nneibor = (unsigned *) malloc(num_of_nodes * sizeof(unsigned));
	for (unsigned i = 0; i < num_of_nodes; ++i) {
		fscanf(fin, "%u", nneibor + i);
	}
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
	unsigned edge_start = 0;
	for (unsigned i = 0; i < num_of_nodes; ++i) {
		h_graph_nodes[i].starting = edge_start;
		h_graph_nodes[i].num_of_edges = nneibor[i];
		edge_start += nneibor[i];
		//h_graph_mask[i] = 0;
		//h_updating_graph_mask[i] = 0;
		//h_graph_visited[i] = 0;
		//h_cost[i] = -1;
	}

	unsigned source = 0;
	//file_name = file_prefix + "-e0.txt";
	//finput = fopen(file_name.c_str(), "r");
	//fscanf(finput, "%u %u", &source, &edge_list_size);
	//fclose(finput);
	//h_graph_mask[source] = 1;
	//h_graph_visited[source] = 1;
	//h_cost[source] = 0;
	//num_lines = edge_list_size / NUM_CORE;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
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
	NUM_THREADS = 256;
#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < edge_list_size; ++i) {
		h_graph_edges[i] = n2s[i];
	}
	free(n1s);
	free(n2s);

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
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

		BFS_kernel(\
				h_graph_nodes,\
				h_graph_mask,\
				h_updating_graph_mask,\
				h_graph_visited,\
				h_graph_edges,\
				h_cost,\
				num_of_nodes,\
				edge_list_size\
				);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
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
		fprintf(stderr, "Error: connot open file %s.\n", file_name.c_str());
		exit(1);
	}
	unsigned bound_i;
	if (NUM_THREADS - 1 != tid) {
		bound_i = num_lines + offset;
	} else {
		bound_i = num_of_nodes;
	}
	for (unsigned index = offset; index < bound_i; ++index) {
		fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
	}
	fclose(fpo);
}
#endif
	//printf("Result stored in result.txt\n");

	// cleanup memory
	free(nneibor);
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
}

void BFS_kernel(\
		Node *h_graph_nodes,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_graph_edges,\
		int *h_cost,\
		unsigned num_of_nodes,\
		int edge_list_size\
		)
{

	//printf("Start traversing the tree\n");
	double start_time = omp_get_wtime();
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;

		omp_set_num_threads(NUM_THREADS);
#pragma omp parallel for 
		for(unsigned int tid = 0; tid < num_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true) {
				h_graph_mask[tid]=false;
				int next_starting = h_graph_nodes[tid].starting + h_graph_nodes[tid].num_of_edges;
//#pragma vector always
				for(int i = h_graph_nodes[tid].starting; \
						i < next_starting; \
						i++)
				{
					int id = h_graph_edges[i];
					if(!h_graph_visited[id])
					{
						h_cost[id]=h_cost[tid]+1;
						h_updating_graph_mask[id]=true;
					}
				}
			}
		}
#pragma omp parallel for
		for(unsigned int tid=0; tid< num_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true) {
				h_graph_mask[tid]=true;
				h_graph_visited[tid]=true;
				stop = false;
				h_updating_graph_mask[tid]=false;
			}
		}
	}
	while(!stop);
	double end_time = omp_get_wtime();
	printf("%d %lf\n", NUM_THREADS, (end_time - start_time));
}
