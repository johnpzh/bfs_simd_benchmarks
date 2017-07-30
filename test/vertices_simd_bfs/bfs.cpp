#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <string>
#include <unistd.h>
#include <hbwmalloc.h>

#define NUM_P_INT 16 // Number of packed integers in one __m512i
#define ALIGNED_BYTES 64
//#define SIZE_INT 4
#define SIZE_INT sizeof(int)
#define OPEN

//#define BUFFER_SIZE_MAX 134217728 // 2^27
//#define BUFFER_SIZE_MAX 4096 //
using std::string;
using std::to_string;

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
		int edge_list_size,\
		int num_omp_threads,\
		unsigned BUFFER_SIZE_MAX,\
		unsigned CHUNK_SIZE \
		);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	start = omp_get_wtime();
	BFSGraph( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
	unsigned int num_of_nodes = 0;
	unsigned int edge_list_size = 0;
	char *input_f;
	int	 num_omp_threads;
	unsigned BUFFER_SIZE_MAX;
	unsigned CHUNK_SIZE;
	
	if(argc!=5){
	num_omp_threads = 1;
	//num_omp_threads = 64;
	//input_f = "/home/zpeng/benchmarks/test/localized_graph/local_graph4096.txt";
	input_f = "/home/zpeng/benchmarks/data/graph4096";
	//BUFFER_SIZE_MAX = 4096;
	BUFFER_SIZE_MAX = 256;
	CHUNK_SIZE = 32768;
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
	BUFFER_SIZE_MAX = atoi(argv[3]);
	CHUNK_SIZE = atoi(argv[4]);
	}
	
	unsigned NUM_CORE = 64;
	omp_set_num_threads(NUM_CORE);
	string file_prefix = input_f;
	string file_name = file_prefix + "-v0.txt";
	FILE *finput = fopen(file_name.c_str(), "r");
	fscanf(finput, "%u", &num_of_nodes);
	fclose(finput);
	unsigned num_lines = num_of_nodes / NUM_CORE;
	//Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*num_of_nodes);
	Node* h_graph_nodes = (Node*) hbw_malloc(sizeof(Node)*num_of_nodes);
	int *h_graph_mask = (int*) _mm_malloc(SIZE_INT*num_of_nodes, ALIGNED_BYTES);
	int *h_updating_graph_mask = (int*) _mm_malloc(SIZE_INT*num_of_nodes, ALIGNED_BYTES);
	int *h_graph_visited = (int*) _mm_malloc(SIZE_INT*num_of_nodes, ALIGNED_BYTES);
	int* h_cost = (int*) _mm_malloc(SIZE_INT*num_of_nodes, ALIGNED_BYTES);
#pragma omp parallel private(file_name, finput)
{
	unsigned thid = omp_get_thread_num();
	file_name = file_prefix + "-v" + to_string(thid) + ".txt";
	finput = fopen(file_name.c_str(), "r");
	if (0 == thid) {
		fscanf(finput, "%u", &num_of_nodes);
	}
	unsigned start, edgeno;
	unsigned offset = thid * num_lines; // here I assume the num_of_nodes can be divided by NUM_CORE
	for (unsigned i = 0; i < num_lines; ++i) {
		fscanf(finput, "%u %u", &start, &edgeno);
		unsigned index = i + offset;
		h_graph_nodes[index].starting = start;
		h_graph_nodes[index].num_of_edges = edgeno;
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

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
	// BFS
	for (unsigned i = 0; i < 9; ++i) {
		num_omp_threads = (unsigned) pow(2, i);
		sleep(10);
		BFS_kernel(\
				h_graph_nodes,\
				h_graph_mask,\
				h_updating_graph_mask,\
				h_graph_visited,\
				h_graph_edges,\
				h_cost,\
				num_of_nodes,\
				edge_list_size,\
				num_omp_threads,\
				BUFFER_SIZE_MAX,\
				CHUNK_SIZE \
				);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", num_omp_threads, now - start);
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
	}
	fclose(time_out);
	
	//Store the result into a file
#ifdef ONEDEBUG
	omp_set_num_threads(NUM_CORE);
	num_lines = num_of_nodes / NUM_CORE;
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
	//free( h_graph_nodes);
	hbw_free( h_graph_nodes);
	_mm_free( h_graph_edges);
	_mm_free( h_graph_mask);
	_mm_free( h_updating_graph_mask);
	_mm_free( h_graph_visited);
	_mm_free( h_cost);
}

///////////////////////////////////////////////////
// BFS Kernel
///////////////////////////////////////////////////
void BFS_kernel(\
		Node *h_graph_nodes,\
		int *h_graph_mask,\
		int *h_updating_graph_mask,\
		int *h_graph_visited,\
		int *h_graph_edges,\
		int *h_cost,\
		unsigned num_of_nodes,\
		int edge_list_size,\
		int num_omp_threads,\
		unsigned BUFFER_SIZE_MAX,\
		unsigned CHUNK_SIZE \
		)
{
	//int k=0;
	const __m512i one_v = _mm512_set1_epi32(1);
	const __m512i minusone_v = _mm512_set1_epi32(-1);
	const __m512i zero_v = _mm512_set1_epi32(0);

	int *id_buffer = (int *) _mm_malloc(SIZE_INT * BUFFER_SIZE_MAX * num_omp_threads, ALIGNED_BYTES);
	int *cost_buffer = (int *) _mm_malloc(SIZE_INT * BUFFER_SIZE_MAX * num_omp_threads, ALIGNED_BYTES);
	unsigned *tops = (unsigned *) _mm_malloc(SIZE_INT * num_omp_threads, ALIGNED_BYTES);
	int thrd;
	unsigned *offsets = (unsigned *) _mm_malloc(SIZE_INT * num_omp_threads, ALIGNED_BYTES);
	for (unsigned i = 0; i < num_omp_threads; ++i) {
		offsets[i] = i * BUFFER_SIZE_MAX;
	}
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
#pragma omp parallel private(thrd)
	{
		thrd = omp_get_thread_num();
		tops[thrd] = 0;
#pragma omp for schedule(dynamic, CHUNK_SIZE) 
//#pragma omp for schedule(static)
#endif 
		for(unsigned int tid = 0; tid < num_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == 1) {
				h_graph_mask[tid]=0;
				int next_starting = h_graph_nodes[tid].starting + h_graph_nodes[tid].num_of_edges;

				for (int i = h_graph_nodes[tid].starting; \
						 i < next_starting; \
						 i += NUM_P_INT) {
					/* Check buffer's size */
					if (tops[thrd] + NUM_P_INT > BUFFER_SIZE_MAX) {
						/* If Buffer would be full, then operate */
						for (unsigned k = 0; k < tops[thrd]; k += NUM_P_INT) {
							if (k + NUM_P_INT <= tops[thrd]) {
								/* Vectoried */
								/* Update those flags */
								__m512i id_v = _mm512_load_epi32(id_buffer + offsets[thrd] + k);
								__m512i visited_v = _mm512_i32gather_epi32(id_v, h_graph_visited, SIZE_INT);
								__mmask16 novisited_mask = _mm512_cmpeq_epi32_mask(visited_v, zero_v);
								_mm512_mask_i32scatter_epi32(h_updating_graph_mask, novisited_mask, id_v, one_v, SIZE_INT);
								_mm512_mask_i32scatter_epi32(h_graph_visited, novisited_mask, id_v, one_v, SIZE_INT);

								/* Update the h_cost */
								__m512i cost_source_v = _mm512_load_epi32(cost_buffer + offsets[thrd] + k);
								__m512i cost_v = _mm512_add_epi32(cost_source_v, one_v);
								_mm512_mask_i32scatter_epi32(h_cost, novisited_mask, id_v, cost_v, SIZE_INT);
							} else {
								/* Serialized */
								for (unsigned j = k; j < tops[thrd]; ++j) {
									int id = id_buffer[offsets[thrd] + j];
									if (!h_graph_visited[id]) {
										h_updating_graph_mask[id] = 1;
										h_graph_visited[id] = 1;
										h_cost[id] = cost_buffer[offsets[thrd] + j] + 1;
									}
								}
							}
						}

						tops[thrd] = 0;
					}
					/* Load to buffer */
					__m512i id_v = _mm512_loadu_si512(h_graph_edges + i);
					_mm512_store_epi32(id_buffer + offsets[thrd] + tops[thrd], id_v);
					__m512i cost_source_v = _mm512_set1_epi32(h_cost[tid]);
					_mm512_store_epi32(cost_buffer + offsets[thrd] + tops[thrd], cost_source_v);
					if (i + NUM_P_INT <= next_starting) {
						tops[thrd] += NUM_P_INT;
					} else {
						tops[thrd] += next_starting - i;
					}
				}
			}
		}

		for (unsigned i = 0; i < tops[thrd]; i += NUM_P_INT) {
			if (i + NUM_P_INT <= tops[thrd]) {
				/* Vectoried */
				/* Update those flags */
				__m512i id_v = _mm512_load_epi32(id_buffer + offsets[thrd] + i);
				__m512i visited_v = _mm512_i32gather_epi32(id_v, h_graph_visited, SIZE_INT);
				__mmask16 novisited_mask = _mm512_cmpeq_epi32_mask(visited_v, zero_v);
				_mm512_mask_i32scatter_epi32(h_updating_graph_mask, novisited_mask, id_v, one_v, SIZE_INT);
				_mm512_mask_i32scatter_epi32(h_graph_visited, novisited_mask, id_v, one_v, SIZE_INT);

				/* Update the h_cost */
				__m512i cost_source_v = _mm512_load_epi32(cost_buffer + offsets[thrd] + i);
				__m512i cost_v = _mm512_add_epi32(cost_source_v, one_v);
				_mm512_mask_i32scatter_epi32(h_cost, novisited_mask, id_v, cost_v, SIZE_INT);
			} else {
				/* Serialized */
				for (unsigned j = i; j < tops[thrd]; ++j) {
					int id = id_buffer[offsets[thrd] + j];
					if (!h_graph_visited[id]) {
						h_updating_graph_mask[id] = 1;
						h_graph_visited[id] = 1;
						h_cost[id] = cost_buffer[offsets[thrd] + j] + 1;
					}
				}
			}
		}
#ifdef OPEN
#pragma omp barrier
#pragma omp for schedule(dynamic, CHUNK_SIZE)
//#pragma omp for schedule(static)
#endif
		for (unsigned i = 0; i < num_of_nodes; i += NUM_P_INT) {
			__m512i id_v = _mm512_load_epi32(h_updating_graph_mask + i);
			__mmask16 updated_v = _mm512_cmpeq_epi32_mask(id_v, one_v);
			if (*((short *)(&updated_v)) != 0) {
				if (stop) {
					stop = 0;
				}
				_mm512_store_epi32(h_graph_mask + i, id_v);
				_mm512_store_epi32(h_updating_graph_mask + i, zero_v);
			}
		}

#ifdef OPEN
	}
#endif
	}
	while(!stop);
#ifdef OPEN
        double end_time = omp_get_wtime();
		printf("%d %lf\n", num_omp_threads, (end_time - start_time));
		//printf("%u %lf\n", BUFFER_SIZE_MAX, (end_time - start_time));
		//printf("%u %lf\n", CHUNK_SIZE, (end_time - start_time));
#endif
	_mm_free(id_buffer);
	_mm_free(cost_buffer);
	_mm_free(tops);
	_mm_free(offsets);
}

