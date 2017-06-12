#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>

#define NO_P_INT 16 // Number of packed integers in one __m512i
#define ALIGNED_BYTES 64
//#define SIZE_INT 4
#define SIZE_INT sizeof(int)
#define OPEN

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
	int *h_graph_mask = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	int *h_updating_graph_mask = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	int *h_graph_visited = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	//int* h_graph_edges = (int*) _mm_malloc(SIZE_INT*edge_list_size, ALIGNED_BYTES);
	//int* h_cost = (int*) _mm_malloc( SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	//if (!(h_graph_nodes \
	//	  && h_graph_mask \
	//	  && h_updating_graph_mask \
	//	  && h_graph_visited \
	//	  && h_graph_edges \
	//	  && h_cost)) {
	//	fprintf(stderr, "@94: cannot malloc the graph.\n");
	//	exit(1);
	//}

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
	int* h_graph_edges = (int*) _mm_malloc(SIZE_INT*edge_list_size, ALIGNED_BYTES);
	//int* h_graph_edges = (int*) _mm_malloc(sizeof(int)*edge_list_size, ALIGNED_BYTES);
	if (!h_graph_edges) {
		fprintf(stderr, "@123: cannot malloc h_graph_edges.\n");
		exit(1);
	}
	for(unsigned int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    


	// allocate mem for the result on host side
	int* h_cost = (int*) _mm_malloc( SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	for(unsigned int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;
	
	//printf("Start traversing the tree\n");
	
	//int k=0;
	const __m512i one_v = _mm512_set1_epi32(1);
	const __m512i minusone_v = _mm512_set1_epi32(-1);
	const __m512i zero_v = _mm512_set1_epi32(0);
	
	int *id_buffer = (int *) _mm_malloc(SIZE_INT * BUFFER_SIZE_MAX, ALIGNED_BYTES);
	int *cost_buffer = (int *) _mm_malloc(SIZE_INT * BUFFER_SIZE_MAX, ALIGNED_BYTES);
	if (!(id_buffer && cost_buffer)) {
		fprintf(stderr, "@158: cannot malloc buffers.\n");
		exit(2);
	}
#ifdef OPEN
        double start_time = omp_get_wtime();
#endif
	int stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = 1;
		unsigned int top = 0;

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
				for (int i = h_graph_nodes[tid].starting; \
						 i < next_starting; \
						 i += NO_P_INT) {
					/* Check buffer's size */
					if (top + NO_P_INT > BUFFER_SIZE_MAX) {
						/* If Buffer would be full, then operate */
#ifdef OPEN
#pragma omp parallel for
#endif
						for (unsigned int k = 0; \
								k < top; \
								k += NO_P_INT) {
							if (k + NO_P_INT <= top) {
								/* Vectoried */
								/* Update those flags */
								__m512i id_v = _mm512_load_epi32(id_buffer + k);
								__m512i visited_v = _mm512_i32gather_epi32(id_v, h_graph_visited, SIZE_INT);
								__mmask16 novisited_mask = _mm512_cmpeq_epi32_mask(visited_v, zero_v);
								_mm512_mask_i32scatter_epi32(h_updating_graph_mask, novisited_mask, id_v, one_v, SIZE_INT);
								_mm512_mask_i32scatter_epi32(h_graph_visited, novisited_mask, id_v, one_v, SIZE_INT);
								stop = 0;

								/* Update the h_cost */
								__m512i cost_source_v = _mm512_load_epi32(cost_buffer + k);
								__m512i cost_v = _mm512_add_epi32(cost_source_v, one_v);
								_mm512_mask_i32scatter_epi32(h_cost, novisited_mask, id_v, cost_v, SIZE_INT);
							} else {
								/* Serialized */
								for (unsigned int j = k; \
										j < top; \
										j++) {
									int id = id_buffer[j];
									if (!h_graph_visited[id]) {
										h_updating_graph_mask[id] = 1;
										h_graph_visited[id] = 1;
										stop = 0;
										h_cost[id] = cost_buffer[j] + 1;
									}
								}
							}
						}

						top = 0;
					}
					/* Load to buffer */
					__m512i id_v = _mm512_loadu_si512(h_graph_edges + i);
					_mm512_store_epi32(id_buffer + top, id_v);
					__m512i cost_source_v = _mm512_set1_epi32(h_cost[tid]);
					_mm512_store_epi32(cost_buffer + top, cost_source_v);
					if (i + NO_P_INT <= next_starting) {
						top += NO_P_INT;
					} else {
						top += next_starting - i;
					}
				}
			}
		}
#ifdef OPEN
#pragma omp parallel for
#endif
		//for (unsigned int i = 0; \
		//	 i < buffer_size; \
		//	 i++) {
		//	int id = id_buffer[i];
		//	int tid = tid_buffer[id];
		//	h_cost[id] = h_cost[tid] + 1;
		//	h_updating_graph_mask[id] = 1;
		//}
		for (unsigned int i = 0; \
			 i < top; \
			 i += NO_P_INT) {
			if (i + NO_P_INT <= top) {
				/* Vectoried */
				/* Update those flags */
				__m512i id_v = _mm512_load_epi32(id_buffer + i);
				__m512i visited_v = _mm512_i32gather_epi32(id_v, h_graph_visited, SIZE_INT);
				__mmask16 novisited_mask = _mm512_cmpeq_epi32_mask(visited_v, zero_v);
				_mm512_mask_i32scatter_epi32(h_updating_graph_mask, novisited_mask, id_v, one_v, SIZE_INT);
				_mm512_mask_i32scatter_epi32(h_graph_visited, novisited_mask, id_v, one_v, SIZE_INT);
				stop = 0;

				/* Update the h_cost */
				__m512i cost_source_v = _mm512_load_epi32(cost_buffer + i);
				__m512i cost_v = _mm512_add_epi32(cost_source_v, one_v);
				_mm512_mask_i32scatter_epi32(h_cost, novisited_mask, id_v, cost_v, SIZE_INT);
			} else {
				/* Serialized */
				for (unsigned int j = i; \
						j < top; \
						j++) {
					int id = id_buffer[j];
					if (!h_graph_visited[id]) {
						h_updating_graph_mask[id] = 1;
						h_graph_visited[id] = 1;
						stop = 0;
						h_cost[id] = cost_buffer[j] + 1;
					}
				}
			}
		}
#ifdef OPEN
#pragma omp parallel for
#endif
		//for(unsigned int tid=0; tid< no_of_nodes ; tid++ )
		//{
		//	if (h_updating_graph_mask[tid] == 1) {
		//		h_graph_mask[tid]=1;
		//		h_graph_visited[tid]=1;
		//		stop = 0;
		//		h_updating_graph_mask[tid]=0;
		//	}
		//}
		//k++;
		//if (!stop) {
		for (unsigned i = 0; i < no_of_nodes; i += NO_P_INT) {
			__m512i id_v = _mm512_load_epi32(h_updating_graph_mask + i);
			_mm512_store_epi32(h_graph_mask + i, id_v);
			_mm512_store_epi32(h_updating_graph_mask + i, zero_v);
		}
		//}
	}
	while(!stop);
#ifdef OPEN
        double end_time = omp_get_wtime();
		printf("%d %lf\n", num_omp_threads, (end_time - start_time));
		//printf("%u %lf\n", BUFFER_SIZE_MAX, (end_time - start_time));
#endif
	//Store the result into a file
	FILE *fpo = fopen("path.txt","w");
	for(unsigned int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	//printf("Result stored in result.txt\n");

	// cleanup memory
	free( h_graph_nodes);
	_mm_free( h_graph_edges);
	_mm_free( h_graph_mask);
	_mm_free( h_updating_graph_mask);
	_mm_free( h_graph_visited);
	_mm_free( h_cost);
	_mm_free( id_buffer);
	//_mm_free( tid_buffer);
	_mm_free( cost_buffer);
}

