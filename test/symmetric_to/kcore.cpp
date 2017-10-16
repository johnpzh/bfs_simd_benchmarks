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
void input(char filename[], unsigned *&graph_heads, unsigned *&graph_ends, unsigned *&graph_degrees) 
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
		graph_degrees[n2]++;
	}

	fclose(fin);
}

	//fname = string(filename) + "_symmetric";
	//fname = string(filename) + "_nohead_symmetric";
	fname = string(filename) + "_nohead";
	fin = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < NEDGES; ++i) {
		unsigned head = graph_heads[i];
		unsigned end = graph_ends[i];
		fprintf(fin, "%u %u\n", head, end);
	}
	fclose(fin);
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
		graph_degrees[end]++;
	}
	fclose(fin);
	//string fname = string(filename) + "_symmetric";
	//fin = fopen(fname.c_str(), "w");
	//for (unsigned i = 0; i < NEDGES; ++i) {
	//	unsigned head = graph_heads[i];
	//	unsigned end = graph_ends[i];
	//	head++;
	//	end++;
	//	fprintf(fin, "%u %u\n", head, end);
	//	fprintf(fin, "%u %u\n", end, head);
	//}
	string fname = string(filename) + "_nohead";
	fin = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < NEDGES; ++i) {
		unsigned head = graph_heads[i];
		unsigned end = graph_ends[i];
		fprintf(fin, "%u %u\n", head, end);
	}
	fclose(fin);
}

int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 1) {
		filename = argv[1];
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_ends;
	unsigned *graph_degrees;
	//unsigned *nneibor;
#ifdef ONESERIAL
	input_serial(filename, graph_heads, graph_ends,graph_degrees);
#else
	input(filename, graph_heads, graph_ends, graph_degrees);
#endif

	// K-core

	// Free memory
	free(graph_heads);
	free(graph_ends);
	free(graph_degrees);

	return 0;
}
