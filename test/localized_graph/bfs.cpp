#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
//#define NUM_THREAD 4
#define OPEN

using std::vector;
using std::sort;
using std::string;
using std::to_string;
using std::endl;
using std::ofstream;


FILE *fp;

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

void BFSGraph(int argc, char** argv);

//void Usage(int argc, char**argv){
void Usage( char**argv){

fprintf(stderr,"Usage: %s <num_threads> <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	BFSGraph( argc, argv);
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
	unsigned int no_of_nodes = 0;
	int edge_list_size = 0;
	char *input_f;
	int	 num_omp_threads;
	
	if(argc!=3){
	//Usage(argc, argv);
	//Usage( argv);
	//exit(0);
	num_omp_threads = 1;
	input_f = "/home/zpeng/benchmarks/data/graph4096";
	} else {
	num_omp_threads = atoi(argv[1]);
	if (!strcmp(argv[2], "4096")) {
		input_f = "/home/zpeng/benchmarks/data/graph4096";
	} else if (!strcmp(argv[2], "16M")) {
		input_f = "/home/zpeng/benchmarks/data/graph16M";
	} else if (!strcmp(argv[2], "128M")) {
		input_f = "/home/zpeng/benchmarks/data/graph128M";
	} else {
		input_f = argv[2];
	}
	}
	
	//printf("Reading File\n");
	//Read in Graph from a file
//	fp = fopen(input_f,"r");
//	if(!fp)
//	{
//		printf("Error Reading graph file\n");
//		return;
//	}
//
//	int source = 0;
//
//	fscanf(fp,"%ud",&no_of_nodes);
//   
//	// allocate host memory
//	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
//	int *h_graph_mask = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
//	int *h_updating_graph_mask = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
//	int *h_graph_visited = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
//
//	int start, edgeno;   
//	// initalize the memory
//	for( unsigned int i = 0; i < no_of_nodes; i++) 
//	{
//		fscanf(fp,"%d %d",&start,&edgeno);
//		h_graph_nodes[i].starting = start;
//		h_graph_nodes[i].no_of_edges = edgeno;
//		h_graph_mask[i]=0;
//		h_updating_graph_mask[i]=0;
//		h_graph_visited[i]=0;
//	}
//
//	//read the source node from the file
//	fscanf(fp,"%d",&source);
//	// source=0; //tesing code line
//
//	//set the source node as true in the mask
//	h_graph_mask[source]=1;
//	h_graph_visited[source]=1;
//
//	fscanf(fp,"%d",&edge_list_size);
//
//	int id,cost;
//	int* h_graph_edges = (int*) _mm_malloc(SIZE_INT*edge_list_size, ALIGNED_BYTES);
//
//	for(unsigned int i=0; i < edge_list_size ; i++)
//	{
//		fscanf(fp,"%d %d",&id, &cost);
//		//fscanf(fp,"%d",&cost);
//		h_graph_edges[i] = id;
//	}
//
//	if(fp)
//		fclose(fp);    
//
//
//	// allocate mem for the result on host side
//	int* h_cost = (int*) _mm_malloc( SIZE_INT*no_of_nodes, ALIGNED_BYTES);
//	for(unsigned int i=0;i<no_of_nodes;i++)
//		h_cost[i]=-1;
//	h_cost[source]=0;

	unsigned NUM_CORE = 64;
	omp_set_num_threads(NUM_CORE);
	string file_prefix = input_f;
	string file_name = file_prefix + "-v0.txt";
	FILE *finput = fopen(file_name.c_str(), "r");
	fscanf(finput, "%u", &no_of_nodes);
	fclose(finput);
	unsigned num_lines = no_of_nodes / NUM_CORE;
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	//int *h_graph_mask = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	//int *h_updating_graph_mask = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	//int *h_graph_visited = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	//int* h_cost = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	int *h_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	int *h_graph_visited = (int*) malloc(sizeof(int)*no_of_nodes);
	int* h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
#pragma omp parallel private(file_name, finput)
{
	unsigned thid = omp_get_thread_num();
	file_name = file_prefix + "-v" + to_string(thid) + ".txt";
	finput = fopen(file_name.c_str(), "r");
	if (0 == thid) {
		fscanf(finput, "%u", &no_of_nodes);
	}
	unsigned start, edgeno;
	unsigned offset = thid * num_lines; // here I assume the no_of_nodes can be divided by NUM_CORE
	for (unsigned i = 0; i < num_lines; ++i) {
		fscanf(finput, "%u %u", &start, &edgeno);
		unsigned index = i + offset;
		h_graph_nodes[index].starting = start;
		h_graph_nodes[index].no_of_edges = edgeno;
		h_graph_mask[index] = 0;
		h_updating_graph_mask[index] = 0;
		h_graph_visited[index] = 0;
		h_cost[index] = -1;
	}
	fclose(finput);
}

	unsigned source;
	file_name = file_prefix + "-e0.txt";
	finput = fopen(file_name.c_str(), "r");
	fscanf(finput, "%u %u", &source, &edge_list_size);
	fclose(finput);
	h_graph_mask[source] = 1;
	h_graph_visited[source] = 1;
	h_cost[source] = 0;
	num_lines = edge_list_size / NUM_CORE;
	//int* h_graph_edges = (int*) _mm_malloc(SIZE_INT*edge_list_size, ALIGNED_BYTES);
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
#pragma omp parallel private(file_name, finput)
{
	unsigned thid = omp_get_thread_num();
	file_name = file_prefix + "-e" + to_string(thid) + ".txt";
	finput = fopen(file_name.c_str(), "r");
	if (0 == thid) {
		fscanf(finput, "%u %u", &source, &edge_list_size);
	}
	unsigned id;
	unsigned cost;
	unsigned offset = thid * num_lines;
	if (NUM_CORE - 1 != thid) {
		for (unsigned i = 0; i < num_lines; ++i) {
			fscanf(finput, "%u %u", &id, &cost);
			unsigned index = i + offset;
			h_graph_edges[index] = id;
		}
	} else {
		for (unsigned i = 0; fscanf(finput, "%u %u", &id, &cost) != EOF; ++i) {
			unsigned index = i + offset;
			h_graph_edges[index] = id;
		}
	}
	fclose(finput);
}

	
	//printf("Start traversing the tree\n");
	
	int k=0;
	vector<unsigned> ordered_nodes_id;
#ifdef OPEN
        double start_time = omp_get_wtime();
#endif
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;

		vector<unsigned> nodes;

#ifdef OPEN
		omp_set_num_threads(num_omp_threads);
#pragma omp parallel for 
#endif 
		for(unsigned int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true) {
				nodes.push_back(tid);
				h_graph_mask[tid]=false;
				int next_starting = h_graph_nodes[tid].starting + h_graph_nodes[tid].no_of_edges;
				//printf("From: %d\t\t", tid);//test
#ifdef OPEN
#pragma vector always
#endif
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

		sort(nodes.begin(), nodes.end());
		for (unsigned i = 0; i < nodes.size(); ++i) {
			//printf("%u\n", nodes[i]);
			ordered_nodes_id.push_back(nodes[i]);
		}
#ifdef OPEN
#pragma omp parallel for
#endif
		for(unsigned int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true) {
				h_graph_mask[tid]=true;
				h_graph_visited[tid]=true;
				stop = false;
				h_updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(!stop);
#ifdef OPEN
        double end_time = omp_get_wtime();
		//printf("%d %lf\n", num_omp_threads, (end_time - start_time));
#endif
	//Store the result into a file
	//FILE *fpo = fopen("path.txt","w");
	//for(unsigned int i=0;i<no_of_nodes;i++)
	//	fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	//fclose(fpo);
	//printf("Result stored in result.txt\n");
	
	vector<Node> ordered_nodes;
	ordered_nodes.resize(no_of_nodes);
	unsigned offset = 0;
	for (unsigned i = 0; i < no_of_nodes; ++i) {
		unsigned old_id = ordered_nodes_id[i];
		ordered_nodes[i].starting = offset;
		ordered_nodes[i].no_of_edges = h_graph_nodes[old_id].no_of_edges;
		offset += ordered_nodes[i].no_of_edges;
	}

	unsigned ordered_source = 0;

	vector<unsigned> ordered_indices;
	ordered_indices.resize(no_of_nodes);
	for (unsigned i = 0; i < ordered_nodes_id.size(); ++i) {
		ordered_indices[ordered_nodes_id[i]] = i;
	}
	
	vector<unsigned> ordered_edges;
	//ordered_edges.resize(edge_list_size);
	for (unsigned i = 0; i < no_of_nodes; ++i) {
		//unsigned starting = ordered_nodes[i].starting;
		unsigned no_of_edges = ordered_nodes[i].no_of_edges;
		unsigned old_id = ordered_nodes_id[i];
		unsigned next_starting = h_graph_nodes[old_id].starting + no_of_edges;
		for (unsigned j = h_graph_nodes[old_id].starting; j < next_starting; ++j) {
			ordered_edges.push_back(ordered_indices[h_graph_edges[j]]);
		}
	}

	//vector<unsigned> ordered_edges;
	//ordered_edges.resize(edge_list_size);
	//for (unsigned i = 0; i < edge_list_size; ++i) {
	//	ordered_edges[i] = ordered_indices[h_graph_edges[i]];
	//}
	//vector<Node> ordered_nodes;
	//ordered_nodes.resize(no_of_nodes);
	//for (unsigned i = 0; i < no_of_nodes; ++i) {
	//	unsigned id = ordered_nodes_id[i];
	//	ordered_nodes[i].starting = h_graph_nodes[id].starting;
	//	ordered_nodes[i].no_of_edges = h_graph_nodes[id].no_of_edges;
	//}
	//unsigned ordered_source = ordered_indices[source];

	//FILE *fout = fopen("ordered_edges.txt", "w");
	//for (unsigned i = 0; i < edge_list_size; ++i) {
	//	fprintf(fout, "%u\n", ordered_edges[i]);
	//}
	//fclose(fout);
	
	//char *output;
	//if (argc == 3) {
	//	if (!strcmp(argv[2], "4096")) {
	//		output = "local_graph4096.txt";
	//	} else if (!strcmp(argv[2], "16M")) {
	//		output = "local_graph16M.txt";
	//	} else if (!strcmp(argv[2], "128M")) {
	//		output = "local_graph128M.txt";
	//	}
	//} else {
	//	output = "local_graph4096.txt";
	//}
	//FILE *fout = fopen(output, "w");
	//fprintf(fout, "%u\n", no_of_nodes);
	//for (unsigned i = 0; i < no_of_nodes; ++i) {
	//	fprintf(fout, "%d %d\n", ordered_nodes[i].starting, ordered_nodes[i].no_of_edges);
	//}
	//fprintf(fout, "\n%d\n", ordered_source);
	//fprintf(fout, "\n%d\n", edge_list_size);
	//for (unsigned i = 0; i < edge_list_size; ++i) {
	//	fprintf(fout, "%u %d\n", ordered_edges[i], 1);
	//}
	//fclose(fout);
	

	//////////////////////////////// Write to File
	omp_set_num_threads(NUM_CORE);
	string prefix = "local_" + string(input_f).substr(string(input_f).find_last_of("/") + 1);
	unsigned totalEdges = edge_list_size;
	unsigned numNodes = no_of_nodes;
	num_lines = numNodes / NUM_CORE;
#pragma omp parallel
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * num_lines;
	string output = prefix + "-v" + to_string(tid) + ".txt";
	ofstream outf(output);
	if (0 == tid) {
		outf << numNodes << endl;
	}
	for (unsigned i = 0; i < num_lines; ++i) {
		unsigned index = i + offset;
		outf << ordered_nodes[index].starting << " " << ordered_nodes[index].no_of_edges << endl;
	}
	outf.close();
}

	num_lines = totalEdges / NUM_CORE;
#pragma omp parallel
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * num_lines;
	string output = prefix + "-e" + to_string(tid) + ".txt";
	ofstream outf(output);
	if (0 == tid) {
		outf << ordered_source << endl;
		outf << totalEdges << endl;
	}
	if (NUM_CORE - 1 != tid) {
		for (unsigned i = 0; i < num_lines; ++i) {
			unsigned index = i + offset;
			outf << ordered_edges[index] << " " << 1 << endl;
		}
	} else {
		for (unsigned i = 0; i + offset < totalEdges; ++i) {
			unsigned index = i + offset;
			outf << ordered_edges[index] << " " << 1 << endl;
		}
	}
	outf.close();
}


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
}

