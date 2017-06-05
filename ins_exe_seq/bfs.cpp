#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
//#define NUM_THREAD 4
#define OPEN
#define SIZE_INT sizeof(int)

//#define BUFFER_SIZE_MAX 134217728 // 2^27
//#define BUFFER_SIZE_MAX 4096 //
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
	num_omp_threads = 1;
	static char add[] = "/home/zpeng/benchmarks/rodinia_3.1/data/bfs/graph4096.txt";
	input_f = add;
	BUFFER_SIZE_MAX = 4096;
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
	int *h_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*no_of_nodes);
	int *h_graph_visited = (int*) malloc(sizeof(int)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=0;
		h_updating_graph_mask[i]=0;
		h_graph_visited[i]=0;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	// source=0; //tesing code line

	//set the source node as true in the mask
	h_graph_mask[source]=1;
	h_graph_visited[source]=1;

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
	
	//int k=0;
	
	//vector<int> id_buffer(BUFFER_SIZE_MAX);
	//int tid_buffer[BUFFER_SIZE_MAX];
	//vector<int> tid_buffer(BUFFER_SIZE_MAX);
	int *id_buffer = (int *) malloc(SIZE_INT * BUFFER_SIZE_MAX);
	int *cost_buffer = (int *) malloc(SIZE_INT * BUFFER_SIZE_MAX);
#ifdef OPEN
        double start_time = omp_get_wtime();
#endif
	int stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = 1;
		unsigned top = 0;

#ifdef OPEN
		omp_set_num_threads(num_omp_threads);
//#pragma omp parallel for 
#endif 
		for(unsigned int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == 1) {
				h_graph_mask[tid]=0;
				int next_starting = h_graph_nodes[tid].starting + h_graph_nodes[tid].no_of_edges;
#ifdef OPEN
//#pragma vector always
#endif
				for(int i = h_graph_nodes[tid].starting; \
						i < next_starting; \
						i++) {
				//	int id = h_graph_edges[i];
				//	if(!h_graph_visited[id])
				//	{
				//		h_cost[id]=h_cost[tid]+1;
				//		h_updating_graph_mask[id]=1;
				//	}
					/* Check buffer's size */
					if (top + 1 > BUFFER_SIZE_MAX) {
						/* If Buffer would be full, then operate */
#ifdef OPEN
#pragma omp parallel for
#endif
						for (unsigned k = 0; k < top; ++k) {
							int id = id_buffer[k];
							if (!h_graph_visited[id]) {
								h_updating_graph_mask[id] = 1;
								h_graph_visited[id] = 1;
								stop = 0;
								h_cost[id] = cost_buffer[k] + 1;
							}
						}
						top = 0;
					}
					/* Load to buffer */
					id_buffer[top] = h_graph_edges[i];
					cost_buffer[top] = h_cost[tid];
					top++;
				}
			}
		}
#ifdef OPEN
#pragma omp parallel for
#endif
		for (unsigned long int i = 0; \
			 i < top; \
			 i++) {
			int id = id_buffer[i];
			if (!h_graph_visited[id]) {
				h_updating_graph_mask[id] = 1;
				h_graph_visited[id] = 1;
				stop = 0;
				h_cost[id] = cost_buffer[i] + 1;
			}
		}
#ifdef OPEN
#pragma omp parallel for
#endif
		for(unsigned int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == 1) {
				h_graph_mask[tid]=1;
				h_updating_graph_mask[tid]=0;
			}
		}
		//k++;
	}
	while(!stop);
#ifdef OPEN
        double end_time = omp_get_wtime();
		//printf("%d %lf\n", num_omp_threads, (end_time - start_time));
		printf("%u %lf\n", BUFFER_SIZE_MAX, (end_time - start_time));
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
	free( id_buffer);
	free( cost_buffer);
}

