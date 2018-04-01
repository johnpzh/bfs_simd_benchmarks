#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <immintrin.h>
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
using std::vector;
using std::to_string;



unsigned nnodes, nedges;
unsigned TILE_WIDTH;

double start;
double now;

void manual_sort(vector<unsigned> &n1v, vector<unsigned> &n2v)
{
	unsigned length = n1v.size();
	vector< vector<unsigned> > n1sv(nnodes);
	int *is_n1_active = (int *) malloc(sizeof(int) * nnodes);
	memset(is_n1_active, 0, sizeof(int) * nnodes);
	for (unsigned i = 0; i < length; ++i) {
		unsigned n1 = n1v[i];
		n1--;
		is_n1_active[n1] = 1;
		n1sv[n1].push_back(n2v[i]);
	}
	unsigned edge_id = 0;
	for (unsigned i = 0; i < nnodes; ++i) {
		if (!is_n1_active[i]) {
			continue;
		}
		for (unsigned j = 0; j < n1sv[i].size(); ++j) {
			n1v[edge_id] = i + 1;
			n2v[edge_id] = n1sv[i][j];
			edge_id++;
		}
	}
	edge_id++;
	free(is_n1_active);
}

void input_untiled(char filename[]) {
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", filename);
		exit(1);
	}

	fscanf(fin, "%u %u", &nnodes, &nedges);
#ifdef ONESYMMETRIC
	nedges *= 2;
#endif
	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	memset(nneibor, 0, nnodes * sizeof(unsigned));
	unsigned *n1s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(nedges * sizeof(unsigned));
	vector< vector<unsigned> > n1sv(nnodes);
#ifdef ONESYMMETRIC
	unsigned bound_i = nedges/2;
#else
	unsigned bound_i = nedges;
#endif
	for (unsigned i = 0; i < bound_i; ++i) {
		unsigned n1;
		unsigned n2;
		//fscanf(fin, "%u %u", &n1, &n2);
		fscanf(fin, "%u %u", &n2, &n1); // Reverse Here!
#ifdef ONESYMMETRIC
		n1sv[n1-1].push_back(n2);
		n1sv[n2-1].push_back(n1);
		nneibor[n1-1]++;
		nneibor[n2-1]++;
#else
		n1--;
		n1sv[n1].push_back(n2);
		nneibor[n1]++;
#endif
		if (i % 10000000 == 0) {
			now = omp_get_wtime();
			printf("time: %lf, got %u 10M edges...\n", now - start, i/10000000);//test
		}
	}
	unsigned edge_id = 0;
	for (unsigned i = 0; i < nnodes; ++i) {
		for (unsigned j = 0; j < n1sv[i].size(); ++j) {
			n1s[edge_id] = i + 1;
			n2s[edge_id] = n1sv[i][j];
			edge_id++;
		}
	}
	printf("Got origin data: %s\n", filename);

	string prefix = string(filename) + "_untiled_reverse";
	unsigned NUM_THREADS = 64;
	unsigned edge_bound = nedges / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS)
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * edge_bound;
	string fname = prefix + "-" + to_string(tid);
	FILE *fout = fopen(fname.c_str(), "w");
	if (0 == tid) {
		fprintf(fout, "%u %u\n", nnodes, nedges);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = nedges;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		fprintf(fout, "%u %u\n", n1s[index], n2s[index]);
	}
	fclose(fout);
}
	printf("Main files done...\n");
	string fname = prefix + "-nneibor";
	FILE *fout = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < nnodes; ++i) {
		fprintf(fout, "%u\n", nneibor[i]);
	}
	// Clean the vectors for saving memory
	fclose(fin);
	fclose(fout);
	free(nneibor);
	free(n1s);
	free(n2s);
}

int main(int argc, char *argv[]) {
	start = omp_get_wtime();
	char *filename;
	if (argc > 1) {
		filename = argv[1];
		//TILE_WIDTH = strtoul(argv[2], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		//TILE_WIDTH = 1024;
	}
	input_untiled(filename);
	return 0;
}
