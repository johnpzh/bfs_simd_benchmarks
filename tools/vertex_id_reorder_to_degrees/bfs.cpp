/*
 * Take 64 xxx_untiled files as input, reorder all vertices according to their degrees, higher degree for smaller ID.
 * After redordered, vertex 0 has the most degrees.
 * Output is 1 file xxx_degreeReordered.
*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <utility>
#include <algorithm>
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
				unsigned *graph_heads,
				unsigned *graph_tails,
				unsigned *nneibor,
				unsigned NNODES,
				unsigned NEDGES,
				char *filename)
{
	puts("Ranking...");
	std::vector< std::pair<unsigned, unsigned> > degree2id;
	for (unsigned v = 0; v < NNODES; ++v) {
		degree2id.emplace_back(nneibor[v] + float(rand()) / RAND_MAX, v);
	}
	std::sort(degree2id.rbegin(), degree2id.rend()); // sort according to degrees.
	std::vector<unsigned> rank(NNODES);
	for (unsigned r = 0; r < NNODES; ++r) {
		rank[degree2id[r].second] = r; // get the rank[v] for every vertex v.
	}

	puts("Writing...");
	string fname = string(filename) + "_degreeReordered";
	FILE *fout = fopen(fname.c_str(), "w");
	fprintf(fout, "%u %u\n", NNODES, NEDGES);
	for (unsigned e_i = 0; e_i < NEDGES; ++e_i) {
		unsigned head = graph_heads[e_i];
		unsigned tail = graph_tails[e_i];
		unsigned new_head = rank[head];
		unsigned new_tail = rank[tail];
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
	puts("Reading data...");
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

	make_up_data(
			n1s,
			n2s,
			nneibor,
			num_of_nodes,
			edge_list_size,
			input_f);




	// cleanup memory
	free(n1s);
	free(n2s);
	free(nneibor);
}

////////////////////////////////////////////////////////////////
// Weighted
void make_up_data_weighted(
				unsigned *graph_heads,
				unsigned *graph_tails,
				unsigned *weights,
				unsigned *nneibor,
				unsigned NNODES,
				unsigned NEDGES,
				char *filename)
{
	puts("Ranking...");
	std::vector< std::pair<unsigned, unsigned> > degree2id;
	for (unsigned v = 0; v < NNODES; ++v) {
		degree2id.emplace_back(nneibor[v] + float(rand()) / RAND_MAX, v);
	}
	std::sort(degree2id.rbegin(), degree2id.rend()); // sort according to degrees.
	std::vector<unsigned> rank(NNODES);
	for (unsigned r = 0; r < NNODES; ++r) {
		rank[degree2id[r].second] = r; // get the rank[v] for every vertex v.
	}

	puts("Writing...");
	string fname = string(filename) + "_degreeReordered";
	FILE *fout = fopen(fname.c_str(), "w");
	fprintf(fout, "%u %u\n", NNODES, NEDGES);
	for (unsigned e_i = 0; e_i < NEDGES; ++e_i) {
		unsigned head = graph_heads[e_i];
		unsigned tail = graph_tails[e_i];
		unsigned wt = weights[e_i];
		unsigned new_head = rank[head];
		unsigned new_tail = rank[tail];
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
	puts("Reading data...");
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

	make_up_data_weighted(
			n1s,
			n2s,
			weights,
			nneibor,
			num_of_nodes,
			edge_list_size,
			input_f);




	// cleanup memory
	free(n1s);
	free(n2s);
	free(weights);
	free(nneibor);
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
