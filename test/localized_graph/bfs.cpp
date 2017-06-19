#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <algorithm>
#include <vector>
//#define NUM_THREAD 4
#define OPEN

using std::vector;
using std::sort;


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
	static char add[] = "/home/zpeng/benchmarks/rodinia_3.1/data/bfs/graph4096.txt";
	input_f = add;
	} else {
	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
	}
	
	//printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%ud",&no_of_nodes);
   
	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	// source=0; //tesing code line

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    


	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(unsigned int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	//printf("Start traversing the tree\n");
	
	int k=0;
	vector<unsigned> ordered_nodes;
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
			ordered_nodes.push_back(nodes[i]);
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
	
	vector<unsigned> ordered_indices;
	ordered_indices.resize(no_of_nodes);
	for (unsigned i = 0; i < ordered_nodes.size(); ++i) {
		ordered_indices[ordered_nodes[i]] = i;
	}
	vector<unsigned> ordered_edges;
	ordered_edges.resize(edge_list_size);
	for (unsigned i = 0; i < edge_list_size; ++i) {
		ordered_edges[i] = ordered_indices[h_graph_edges[i]];
	}

	FILE *fout = fopen("ordered_edges.txt", "w");
	for (unsigned i = 0; i < edge_list_size; ++i) {
		fprintf(fout, "%u\n", ordered_edges[i]);
	}
	fclose(fout);

	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
}

