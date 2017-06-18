#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <algorithm>
//#define NUM_THREAD 4
#define OPEN

using std::vector;
using std::sort;

//#define BUFFER_SIZE_MAX 16
unsigned BUFFER_SIZE_MAX;


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
	
	if(argc!=4){
	//Usage(argc, argv);
	//Usage( argv);
	//exit(0);
	num_omp_threads = 64;
	//num_omp_threads = 1;
	static char add[] = "/home/zpeng/benchmarks/rodinia_3.1/data/bfs/graph4096.txt";
	input_f = add;
	BUFFER_SIZE_MAX = 64;
	} else {
	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
	BUFFER_SIZE_MAX = atoi(argv[3]);
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
	
	//vector<unsigned> nodes;
	//nodes.resize(4096);
	int k=0;
	int **id_buffer = (int **) malloc(sizeof(int *) * num_omp_threads);
	int **cost_buffer = (int **) malloc(sizeof(int *) * num_omp_threads);
	for (unsigned i = 0; i < num_omp_threads; ++i) {
		id_buffer[i] = (int *) malloc(sizeof(int) * BUFFER_SIZE_MAX);
		cost_buffer[i] = (int *) malloc(sizeof(int) * BUFFER_SIZE_MAX);
	}
	unsigned top = 0;
#ifdef OPEN
        double start_time = omp_get_wtime();
#endif
	bool stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = true;
		//nodes.clear();
		//printf("\n*******\n");//test

#ifdef OPEN
		omp_set_num_threads(num_omp_threads);
#pragma omp parallel private(top)
		{
		top = 0;
		int thrd = omp_get_thread_num();
#pragma omp for schedule(dynamic)
//#pragma omp for
#endif 
		for(unsigned int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true) {
				//nodes.push_back(tid);
				h_graph_mask[tid]=false;
				int next_starting = h_graph_nodes[tid].starting + h_graph_nodes[tid].no_of_edges;

				for(int i = h_graph_nodes[tid].starting; \
						i < next_starting; \
						i++)
				{
					/* Check buffer's size, if the buffer gonna to be full */
					if (top + 1 > BUFFER_SIZE_MAX) {
						for (unsigned j = 0; j < top; j++) {
							int id = id_buffer[thrd][j];
							if (!h_graph_visited[id]) {
								h_updating_graph_mask[id] = true;
								h_graph_visited[id] = true;
								//stop = false;
								h_cost[id] = cost_buffer[thrd][j] + 1;
								//printf("@182:%d, ", id);//test
							}
						}
						//printf("\n");//test
						top = 0;
					}

					/* Load to buffer */
					id_buffer[thrd][top] = h_graph_edges[i];
					cost_buffer[thrd][top] = h_cost[tid];
					top++;
				}
			}
		}
#ifdef OPEN
//#pragma omp parallel for firstprivate(top)
//#pragma omp for schedule(dynamic)
//#pragma omp for
#endif
		for (unsigned i = 0; i < top; i++) {
			//int thrd = omp_get_thread_num();
			int id = id_buffer[thrd][i];
			if (!h_graph_visited[id]) {
				h_updating_graph_mask[id] = true;
				h_graph_visited[id] = true;
				//stop = false;
				h_cost[id] = cost_buffer[thrd][i] + 1;
				//printf("@209:%d, ", id);//test
			}
		}
		//printf("\n");//test
#pragma omp barrier

//#pragma omp master
//		{
//		printf("\n======= %d ======= nodes.size:%lu \n", k, nodes.size());//test
//		sort(nodes.begin(), nodes.end());
//		for (unsigned i = 0; i < nodes.size(); ++i) {
//			printf("'%u' ", nodes[i]);
//		}
//		printf("\n");//test
//		}

#ifdef OPEN
#pragma omp for schedule(dynamic)
//#pragma omp for
#endif
		for(unsigned int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true) {
				//printf("!%u ", tid);//test
				h_graph_mask[tid]=true;
				//h_graph_visited[tid]=true;
				//stop = false;
				if (stop) {
					stop = false;
				}
				h_updating_graph_mask[tid]=false;
			}
		}
#pragma omp master
		k++;
#ifdef OPEN
		}
#endif
	}
	while(!stop);
#ifdef OPEN
        double end_time = omp_get_wtime();
		//printf("%d %lf\n", num_omp_threads, (end_time - start_time));
#endif
	//Store the result into a file
	FILE *fpo = fopen("path.txt","w");
	for(unsigned int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	//printf("Result stored in result.txt\n");

	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	for (unsigned i = 0; i < num_omp_threads; ++i) {
		free(id_buffer[i]);
		free(cost_buffer[i]);
	}
	free(id_buffer);
	free(cost_buffer);
}

