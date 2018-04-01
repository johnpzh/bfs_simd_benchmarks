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
void BFS_kernel(
		Node *h_graph_nodes,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		int *h_graph_visited,
		unsigned *h_graph_edges,
		int *h_cost,
		unsigned num_of_nodes,
		unsigned edge_list_size,
		unsigned *vertex_map,
		unsigned &top_index);



void make_up_data(
				unsigned *vertex_map,
				unsigned *graph_heads,
				unsigned *graph_tails,
				unsigned NNODES,
				unsigned NEDGES,
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

///////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph
///////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
	unsigned int num_of_nodes = 0;
	unsigned edge_list_size = 0;
	char *input_f;
	
	if(argc > 1){
	//Usage(argc, argv);
	//Usage( argv);
	//exit(0);
	//NUM_THREADS = 1;
	input_f = argv[1];
	} else {
	//NUM_THREADS = atoi(argv[1]);
	printf("Usage: ./bfs <data_file>\n");
	exit(1);
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
	printf("BFSing...\n");

	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*num_of_nodes);
	int *h_graph_mask = (int*) malloc(sizeof(int)*num_of_nodes);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*num_of_nodes);
	int *h_graph_visited = (int*) malloc(sizeof(int)*num_of_nodes);
	int* h_cost = (int*) malloc(sizeof(int)*num_of_nodes);
	unsigned edge_start = 0;
	for (unsigned i = 0; i < num_of_nodes; ++i) {
		h_graph_nodes[i].starting = edge_start;
		h_graph_nodes[i].num_of_edges = nneibor[i];
		edge_start += nneibor[i];
	}

	unsigned source = 0;
	unsigned* h_graph_edges = (unsigned*) malloc(sizeof(unsigned)*edge_list_size);
	NUM_THREADS = 256;
#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < edge_list_size; ++i) {
		h_graph_edges[i] = n2s[i];
	}

	// Map a vertex index to its new index: vertex_map[old] = new;
	unsigned *vertex_map = (unsigned *) malloc(num_of_nodes * sizeof(unsigned));
#pragma omp parallel for num_threads(64)
	for (unsigned i = 0; i < num_of_nodes; ++i) {
		vertex_map[i] = (unsigned) -1;
	}
	unsigned top_index = 0;
	vertex_map[source] = top_index++;

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned run_count = 7;
#else
	unsigned run_count = 7;
#endif
	// BFS
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

		BFS_kernel(
				h_graph_nodes,
				h_graph_mask,
				h_updating_graph_mask,
				h_graph_visited,
				h_graph_edges,
				h_cost,
				num_of_nodes,
				edge_list_size,
				vertex_map,
				top_index);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);

	make_up_data(
			vertex_map,
			n1s,
			n2s,
			num_of_nodes,
			edge_list_size,
			input_f);




	// cleanup memory
	free(n1s);
	free(n2s);
	free(nneibor);
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
}

////////////////////////////////////////////////////////////////
// Weighted
void make_up_data_weighted(
				unsigned *vertex_map,
				unsigned *graph_heads,
				unsigned *graph_tails,
				unsigned *weights,
				unsigned NNODES,
				unsigned NEDGES,
				char *filename)
{
	puts("Writing...");
	string fname = string(filename) + "_reorder";
	FILE *fout = fopen(fname.c_str(), "w");
	fprintf(fout, "%u %u\n", NNODES, NEDGES);
	for (unsigned e_i = 0; e_i < NEDGES; ++e_i) {
		unsigned head = graph_heads[e_i];
		unsigned tail = graph_tails[e_i];
		unsigned wt = weights[e_i];
		unsigned new_head = vertex_map[head];
		unsigned new_tail = vertex_map[tail];
		++new_head;
		++new_tail;
		fprintf(fout, "%u %u %u\n", new_head, new_tail, wt);
	}
	puts("Done.");
}
void BFSGraph_weighted( int argc, char** argv) 
{
	unsigned int num_of_nodes = 0;
	unsigned edge_list_size = 0;
	char *input_f;
	
	if(argc > 1){
	//Usage(argc, argv);
	//Usage( argv);
	//exit(0);
	//NUM_THREADS = 1;
	input_f = argv[1];
	} else {
	//NUM_THREADS = atoi(argv[1]);
	printf("Usage: ./bfs <data_file>\n");
	exit(1);
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
	unsigned *weights = (unsigned *) malloc(edge_list_size * sizeof(unsigned));
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
			unsigned wt;
			fscanf(fin, "%u%u%u", &n1, &n2, &wt);
			n1--;
			n2--;
			n1s[index] = n1;
			n2s[index] = n2;
			weights[index] = wt;
		}
	} else {
		for (unsigned i = 0; i + offset < edge_list_size; ++i) {
			unsigned index = i + offset;
			unsigned n1;
			unsigned n2;
			unsigned wt;
			fscanf(fin, "%u%u%u", &n1, &n2, &wt);
			n1--;
			n2--;
			n1s[index] = n1;
			n2s[index] = n2;
			weights[index] = wt;
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
	printf("BFSing...\n");

	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*num_of_nodes);
	int *h_graph_mask = (int*) malloc(sizeof(int)*num_of_nodes);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*num_of_nodes);
	int *h_graph_visited = (int*) malloc(sizeof(int)*num_of_nodes);
	int* h_cost = (int*) malloc(sizeof(int)*num_of_nodes);
	unsigned edge_start = 0;
	for (unsigned i = 0; i < num_of_nodes; ++i) {
		h_graph_nodes[i].starting = edge_start;
		h_graph_nodes[i].num_of_edges = nneibor[i];
		edge_start += nneibor[i];
	}

	unsigned source = 0;
	unsigned* h_graph_edges = (unsigned*) malloc(sizeof(unsigned)*edge_list_size);
	NUM_THREADS = 256;
#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < edge_list_size; ++i) {
		h_graph_edges[i] = n2s[i];
	}

	// Map a vertex index to its new index: vertex_map[old] = new;
	unsigned *vertex_map = (unsigned *) malloc(num_of_nodes * sizeof(unsigned));
#pragma omp parallel for num_threads(64)
	for (unsigned i = 0; i < num_of_nodes; ++i) {
		vertex_map[i] = (unsigned) -1;
	}
	unsigned top_index = 0;
	vertex_map[source] = top_index++;

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned run_count = 7;
#else
	unsigned run_count = 7;
#endif
	// BFS
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

		BFS_kernel(
				h_graph_nodes,
				h_graph_mask,
				h_updating_graph_mask,
				h_graph_visited,
				h_graph_edges,
				h_cost,
				num_of_nodes,
				edge_list_size,
				vertex_map,
				top_index);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);

	make_up_data_weighted(
			vertex_map,
			n1s,
			n2s,
			weights,
			num_of_nodes,
			edge_list_size,
			input_f);




	// cleanup memory
	free(n1s);
	free(n2s);
	free(weights);
	free(nneibor);
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
}
// End Weighted
////////////////////////////////////////////////////////////

void add_mask2map(
			unsigned *vertex_map,
			unsigned &top_index,
			int *h_graph_mask,
			unsigned NNODES)
{
	unsigned old = top_index;
	for (unsigned i = 0; i < NNODES; ++i) {
		if (!h_graph_mask[i]) {
			continue;
		}
		vertex_map[i] = top_index++;
	}
	printf("added: %u, got: %u (/%u)\n", top_index - old, top_index, NNODES);
}

void add_remainder2map(
		unsigned *vertex_map,
		unsigned &top_index,
		unsigned NNODES)
{
	for (unsigned i =0; i < NNODES; ++i) {
		if ((unsigned) -1 == vertex_map[i]) {
			vertex_map[i] = top_index++;
		}
	}
}

void BFS_kernel(
		Node *h_graph_nodes,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		int *h_graph_visited,
		unsigned *h_graph_edges,
		int *h_cost,
		unsigned num_of_nodes,
		unsigned edge_list_size,
		unsigned *vertex_map,
		unsigned &top_index)
{

	//printf("Start traversing the tree\n");
	double start_time = omp_get_wtime();
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = 1;

		omp_set_num_threads(NUM_THREADS);
//#pragma omp parallel for 
#pragma omp parallel for schedule(dynamic, 512)
		for(unsigned int tid = 0; tid < num_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == 1) {
				h_graph_mask[tid]=0;
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
						h_updating_graph_mask[id]=1;
					}
				}
			}
		}
		add_mask2map(
				vertex_map,
				top_index,
				h_updating_graph_mask,
				num_of_nodes);
//#pragma omp parallel for
#pragma omp parallel for schedule(dynamic, 512)
		for(unsigned int tid=0; tid< num_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == 1) {
				h_graph_mask[tid]=1;
				h_graph_visited[tid]=1;
				stop = 0;
				h_updating_graph_mask[tid]=0;
			}
		}
	} while(!stop);

	add_remainder2map(
		vertex_map,
		top_index,
		num_of_nodes);

	double end_time = omp_get_wtime();
	printf("%d %lf\n", NUM_THREADS, (end_time - start_time));
}

///////////////////////////////////////////////////////////////////////////////
// Main Program
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	start = omp_get_wtime();
#ifdef WEIGHTED
	BFSGraph_weighted( argc, argv );
#else
	BFSGraph( argc, argv);
#endif
}
