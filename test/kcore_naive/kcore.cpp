#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#include <algorithm>
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;
using std::vector;

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned KCORE;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


//void input(char filename[], unsigned *&graph_heads, unsigned *&graph_ends, unsigned *&nneibor) 
void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_ends, 
		unsigned *&graph_degrees)
		//(vector<vector<unsigned>> &graph_neighbors) 
{
	//printf("data: %s\n", filename);
	string prefix = string(filename) + "_untiled";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	fclose(fin);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) malloc(NNODES * sizeof(unsigned));
	//graph_neighbors.resize(NNODES);
	memset(graph_degrees, 0, NNODES * sizeof(unsigned));
	NUM_THREADS = 64;
	unsigned edge_bound = NEDGES / NUM_THREADS;
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
		fscanf(fin, "%u %u\n", &NNODES, &NEDGES);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_ends[index] = n2;
		graph_degrees[n1]++;
		//graph_degrees[n2]++;
//#pragma omp critical
//		{
//		graph_neighbors[n1].push_back(n2);
//		graph_neighbors[n2].push_back(n1);
//		}
	}

	fclose(fin);
}
}

void input_serial(char filename[], unsigned *&graph_heads, unsigned *&graph_ends, unsigned *&graph_degrees)
{
	//printf("data: %s\n", filename);
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", filename);
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) calloc(NNODES, sizeof(unsigned));
	for (unsigned i = 0; i < NEDGES; ++i) {
		unsigned head;
		unsigned end;
		fscanf(fin, "%u %u", &head, &end);
		--head;
		--end;
		graph_heads[i] = head;
		graph_ends[i] = end;
		graph_degrees[head]++;
		//graph_degrees[end]++;
	}
	fclose(fin);
}

void print(unsigned *graph_cores) {
	FILE *foutput = fopen("ranks.txt", "w");
	unsigned kc = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		fprintf(foutput, "%u: %u\n", i+1, graph_cores[i]);
		if (kc < graph_cores[i]) {
			kc = graph_cores[i];
		}
	}
	fprintf(foutput, "kc: %u, KCORE: %u\n", kc, KCORE);
}
unsigned test_count = 0;//test
void kcore_kernel(
				unsigned *graph_heads, 
				unsigned *graph_ends,
				unsigned *graph_degrees,
				//const vector<vector<unsigned>> &graph_neighbors,
				int *graph_updating_active, 
				//const unsigned &node_i_start,
				//const unsigned &node_i_bound,
				const unsigned &edge_i_start, 
				const unsigned &edge_i_bound,
				unsigned *graph_cores)
				//(int &stop)
{
//	int has_remove = 1;
//	while (has_remove) {
//		//double ts = omp_get_wtime();
//		has_remove = 0;
//#pragma omp parallel for
//		for (unsigned i = node_i_start; i < node_i_bound; ++i) {
//			if (graph_degrees[i]) {
//				stop = 0;
//				if(graph_degrees[i] < KCORE) {
//					graph_updating_active[i] = 1;
//					graph_degrees[i] = 0;
//					graph_cores[i] = KCORE - 1;
//					//test_count++;//test
//					has_remove = 1;
//				}
//			}
//		}
		//double ts2 = omp_get_wtime();
		//printf("time for nodes: %lf\n", ts2 - ts);

#pragma omp parallel for
	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		unsigned end = graph_ends[edge_i];
		if (graph_updating_active[head] && graph_degrees[end]) {
			graph_degrees[end]--;
			if (!graph_degrees[end]) {
				graph_cores[end] = KCORE - 1;
				test_count++;//test
			}
		}
		//if (graph_updating_active[end] && graph_degrees[head]) {
		//	graph_degrees[head]--;
		//	if (!graph_degrees[head]) {
		//		graph_cores[head] = KCORE - 1;
		//		test_count++;//test
		//	}
		//}
	}
		//for (unsigned vertex_i = 0; vertex_i < NNODES; ++vertex_i) {
		//	if (!graph_updating_active[vertex_i]) {
		//		continue;
		//	}
		//	for (unsigned neibr_i = 0; neibr_i < graph_neighbors[vertex_i].size(); ++neibr_i) {
		//		unsigned neighbor = graph_neighbors[vertex_i][neibr_i];
		//		if (0 == graph_degrees[neighbor]) {
		//			continue;
		//		}
		//		graph_degrees[neighbor]--;
		//		if (!graph_degrees[neighbor]) {
		//			graph_cores[neighbor] = KCORE - 1;
		//			//test_count++;//test
		//		}
		//	}
		//}
		//double ts3 = omp_get_wtime();
		//printf("time for edges: %lf\n", ts3 - ts2);//test
		//memset(graph_updating_active, 0, NNODES * sizeof(int));
	//}
}
void kcore(
		unsigned *graph_heads, 
		unsigned *graph_ends, 
		unsigned *graph_degrees,
		//const vector<vector<unsigned>> &graph_neighbors,
		int *graph_updating_active,
		unsigned *graph_cores)
{
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	int stop = 0;
	test_count = 0;
	while (!stop) {
		stop = 1;
		int has_remove = 1;
		KCORE++;
		while (has_remove) {
			double ts = omp_get_wtime();
			has_remove = 0;
//#pragma omp parallel for
			for (unsigned i = 0; i < NNODES; ++i) {
				if (graph_degrees[i]) {
					stop = 0;
					if(graph_degrees[i] < KCORE) {
						graph_updating_active[i] = 1;
						graph_degrees[i] = 0;
						graph_cores[i] = KCORE - 1;
						test_count++;//test
						has_remove = 1;
					}
				}
			}
			double ts2 = omp_get_wtime();
			//printf("time for vertices: %lf\n", ts2 - ts);//test
			kcore_kernel(
					graph_heads, 
					graph_ends, 
					graph_degrees,
					//graph_neighbors,
					graph_updating_active, 
					//0,
					//NNODES,
					0, 
					NEDGES,
					graph_cores);
//#pragma omp parallel for
			//for (unsigned edge_i = 0; edge_i < NEDGES; ++edge_i) {
			//	unsigned head = graph_heads[edge_i];
			//	unsigned end = graph_ends[edge_i];
			//	if (graph_updating_active[head] && graph_degrees[end]) {
			//		graph_degrees[end]--;
			//		if (!graph_degrees[end]) {
			//			graph_cores[end] = KCORE - 1;
			//			test_count++;//test
			//		}
			//	}
			//}
		//if (graph_updating_active[end] && graph_degrees[head]) {
		//	graph_degrees[head]--;
		//	if (!graph_degrees[head]) {
		//		graph_cores[head] = KCORE - 1;
		//		test_count++;//test
		//	}
		//}
	//}
			//printf("time for edges: %lf\n", omp_get_wtime() - ts2);
			//(stop);
			memset(graph_updating_active, 0, NNODES * sizeof(int));
		}
		printf("test_count: %u, KCORE: %u\n", test_count, KCORE);//test
		//memset(graph_updating_active, 0, NNODES * sizeof(int));
		//if (!stop) {
		//	KCORE++;
		//} else {
		//	KCORE -= 2;
		//}
	}
	KCORE -= 2;

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
}


int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 1) {
		filename = argv[1];
	} else {
		//filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		filename = "/home/zpeng/benchmarks/data/skitter/out.skitter";
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_ends;
	unsigned *graph_degrees;
	//vector< vector<unsigned> > graph_neighbors;
	//unsigned *nneibor;
#ifdef ONESERIAL
	//input_serial("/home/zpeng/benchmarks/data/fake/data.txt", graph_heads, graph_ends, graph_degrees);
	//input_serial("/home/zpeng/benchmarks/data/fake/mun_twitter", graph_heads, graph_ends,graph_degrees);
	input_serial("/home/zpeng/benchmarks/data/zebra/out.zebra_sym", graph_heads, graph_ends,graph_degrees);
#else
	input(
		filename, 
		graph_heads, 
		graph_ends, 
		graph_degrees);
		//(graph_neighbors);
#endif
	//vector<unsigned> degrees_list(graph_degrees, graph_degrees + NNODES);
	//std::sort(degrees_list.begin(), degrees_list.end());
	//for (unsigned i = 0; i < NNODES; ++i) {
	//	printf("%u: %u\n", i, degrees_list[i]);//test
	//}
	//exit(1);

	// K-core
	int *graph_updating_active = (int *) malloc(NNODES * sizeof(int));
	unsigned *graph_cores = (unsigned *) malloc(NNODES * sizeof(unsigned));
	unsigned *graph_degrees_bak = (unsigned *) malloc(NNODES * sizeof(unsigned));
	memcpy(graph_degrees_bak, graph_degrees, NNODES * sizeof(unsigned));
	
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned run_count = 1;
	printf("Start K-core...\n");
#else
	unsigned run_count = 9;
#endif
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		for (unsigned k = 0; k < NNODES; ++k) {
		}
		memset(graph_updating_active, 0, NNODES * sizeof(int));
		for (unsigned k = 0; k < NNODES; ++k) {
			graph_cores[k] = 0;
		}
		KCORE = 0;
		memcpy(graph_degrees, graph_degrees_bak, NNODES * sizeof(unsigned));
		//sleep(10);
		kcore(
			graph_heads, 
			graph_ends, 
			graph_degrees,
			//graph_neighbors,
			graph_updating_active,
			graph_cores);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);
#ifdef ONEDEBUG
	print(graph_cores);
#endif

	// Free memory
	free(graph_heads);
	free(graph_ends);
	free(graph_degrees);
	free(graph_degrees_bak);
	free(graph_updating_active);
	free(graph_cores);

	return 0;
}
