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
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
using std::to_string;

#define DUMP 0.85

int nnodes, nedges;
unsigned NUM_THREADS;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

void page_rank(unsigned *n1s, unsigned *n2s, unsigned *nneibor, float *rank, float *sum);
void print(float *rank);

void input(char filename[]) {
	printf("input data: %s\n", filename);
	string prefix = string(filename) + "_untiled";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	fscanf(fin, "%u %u", &nnodes, &nedges);
	fclose(fin);
	//memset(nneibor, 0, nnodes * sizeof(unsigned));
	//for (unsigned i = 0; i < nnodes; ++i) {
	//	grah.nneibor[i] = 0;
	//}
	unsigned *n1s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(nedges * sizeof(unsigned));
	NUM_THREADS = 64;
	unsigned edge_bound = nedges / NUM_THREADS;
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
		fscanf(fin, "%u %u\n", &nnodes, &nedges);
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
		for (unsigned i = 0; i + offset < nedges; ++i) {
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

	prefix = string(filename);
	fname = prefix + "_nohead";
	printf("no head version starts...\n");
	FILE *fout = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < nedges; ++i) {
		fprintf(fout, "%u %u\n", n1s[i], n2s[i]);
		if (i % 10000000 == 0) {
			printf(".");
		}
	}
	printf("\n");
	fclose(fout);

	printf("transpose version starts...\n");
	fname = prefix + "_transpose";
	fout = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < nedges; ++i) {
		fprintf(fout, "%u %u\n", n2s[i], n1s[i]);
		if (i % 10000000 == 0) {
			printf(".");
		}
	}
	printf("\n");
	fclose(fout);


	free(n1s);
	free(n2s);
}

int main(int argc, char *argv[]) {
	start = omp_get_wtime();
	char *filename;
	if (argc > 1) {
		filename = argv[1];
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec-relationships.txt";
	}
	input(filename);
	return 0;
}
