#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>

#include <immintrin.h>

#define NO_P_INT 16 // Number of packed integers in one __m512i
#define ALIGNED_BYTES 64
#define SIZE_INT sizeof(int)
//#define NUM_THREAD 4
#define OPEN

using std::string;
using std::to_string;


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
	
	if (argc!=3){
	//Usage(argc, argv);
	//Usage( argv);
	//exit(0);
	num_omp_threads = 1;
	//static char add[] = "/home/zpeng/benchmarks/rodinia_3.1/data/bfs/graph16M.txt";
	input_f = "/home/zpeng/benchmarks/data/graph4096";
	} else {
	num_omp_threads = atoi(argv[1]);
	input_f = argv[2];
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
	int *h_graph_mask = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	int *h_updating_graph_mask = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	int *h_graph_visited = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
	int* h_cost = (int*) _mm_malloc(SIZE_INT*no_of_nodes, ALIGNED_BYTES);
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
	int* h_graph_edges = (int*) _mm_malloc(SIZE_INT*edge_list_size, ALIGNED_BYTES);
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
	
	//int k=0;
	const __m512i one_v = _mm512_set1_epi32(1);
	const __m512i minusone_v = _mm512_set1_epi32(-1);
	const __m512i zero_v = _mm512_set1_epi32(0);
#ifdef OPEN
        double start_time = omp_get_wtime();
#endif
	int stop;
	do
	{
		//if no thread changes this value then the loop stops
		stop = 1;

#ifdef OPEN
		omp_set_num_threads(num_omp_threads);
#pragma omp parallel for 
#endif 
		for(unsigned int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == 1) {
				h_graph_mask[tid]=0;
				//for(int i=h_graph_nodes[tid].starting; \
				//		i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); \
				//		i++)
				//{
				//	int id = h_graph_edges[i];
				//	if(!h_graph_visited[id])
				//	{
				//		h_cost[id]=h_cost[tid]+1;
				//		h_updating_graph_mask[id]=true;
				//	}
				//}

				int next_starting = h_graph_nodes[tid].starting + h_graph_nodes[tid].no_of_edges;
#ifdef OPEN
//#pragma vector always
#endif
				for(int i = h_graph_nodes[tid].starting; \
						i < next_starting; \
						i += NO_P_INT) {
					//if (h_graph_nodes[tid].no_of_edges < NO_P_INT ) {
					__m512i edge_i_v = _mm512_set_epi32(\
							i+15, i+14, i+13, i+12, i+11, i+10, i+9, i+8, \
							i+7,  i+6,  i+5,  i+4,  i+3,  i+2,  i+1,  i+0);
					__m512i n_s_v = _mm512_set1_epi32(next_starting);
					__mmask16 lt_mask = _mm512_cmplt_epi32_mask(edge_i_v, n_s_v);
					__m512i id_v = _mm512_mask_loadu_epi32(minusone_v, lt_mask, h_graph_edges + i);
					__m512i visited_v = _mm512_mask_i32gather_epi32(one_v, lt_mask, id_v, h_graph_visited, sizeof(int));
					//}
					//__m512i id_v = _mm512_load_epi32(h_graph_edges + i);
					//__m512i visited_v = _mm512_i32gather_epi32(id_v, h_graph_visited, sizeof(int));
					__mmask16 novisited_mask = _mm512_cmpeq_epi32_mask(visited_v, zero_v);
					_mm512_mask_i32scatter_epi32(h_updating_graph_mask, novisited_mask, id_v, one_v, sizeof(int));
					
					//__m512i cost_v = _mm512_mask_i32gather_epi32(zero_v, novisited_mask, id_v, h_cost, sizeof(int));
					__m512i cost_source_v = _mm512_set1_epi32(h_cost[tid]);
					__m512i cost_v = _mm512_mask_add_epi32(cost_source_v, novisited_mask, cost_source_v, one_v);
					_mm512_mask_i32scatter_epi32(h_cost, novisited_mask, id_v, cost_v, sizeof(int));
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
		for (unsigned tid = 0; tid < no_of_nodes; tid += NO_P_INT) {
			if (tid + NO_P_INT <= no_of_nodes) {
				__m512i updating_mask_v = _mm512_load_epi32(h_updating_graph_mask + tid);
				__mmask16 true_updating_mask_mask = _mm512_cmpeq_epi32_mask(updating_mask_v, one_v);
				_mm512_mask_store_epi32(h_graph_mask + tid, true_updating_mask_mask, one_v);
				_mm512_mask_store_epi32(h_graph_visited + tid, true_updating_mask_mask, one_v);
				short *updating_flag = (short *) (&true_updating_mask_mask);
				if (*updating_flag) {
					stop = 0;
				}
				//_mm512_store_epi32(h_updating_graph_mask + tid, zero_v);
				_mm512_mask_store_epi32(h_updating_graph_mask + tid, true_updating_mask_mask, zero_v);
			} else {
				for (unsigned i = tid; i < no_of_nodes; i++) {
					if (h_updating_graph_mask[i] == 1) {
						h_graph_mask[i]=1;
						h_graph_visited[i]=1;
						stop = 0;
						h_updating_graph_mask[i]=0;
					}
				}
			}
		}
		//printf("@225\n");//test
		//k++;
	}
	while(!stop);
#ifdef OPEN
        double end_time = omp_get_wtime();
		printf("%d %lf\n", num_omp_threads, (end_time - start_time));
#endif
	//Store the result into a file
	//FILE *fpo = fopen("path.txt","w");
	//for(unsigned int i=0;i<no_of_nodes;i++)
	//	fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	//fclose(fpo);

	omp_set_num_threads(NUM_CORE);
	num_lines = no_of_nodes / NUM_CORE;
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
	//printf("Result stored in result.txt\n");

	// cleanup memory
	//free( h_graph_nodes);
	//free( h_graph_edges);
	//free( h_graph_mask);
	//free( h_updating_graph_mask);
	//free( h_graph_visited);
	//free( h_cost);
	free( h_graph_nodes);
	_mm_free( h_graph_edges);
	_mm_free( h_graph_mask);
	_mm_free( h_updating_graph_mask);
	_mm_free( h_graph_visited);
	_mm_free( h_cost);
}

