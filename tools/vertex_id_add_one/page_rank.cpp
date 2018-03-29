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

unsigned NNODES, NEDGES;
unsigned TILE_WIDTH;

double start;
double now;


void input_untiled(char filename[]) {
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", filename);
		exit(1);
	}

	//fscanf(fin, "%u %u", &NNODES, &NEDGES);
	fscanf(fin, "%*s");
#ifdef ONESYMMETRIC
	NEDGES *= 2;
#endif
	unsigned *n1s = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	unsigned *weits = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	vector< vector<unsigned> > n1sv(NNODES);
	vector< vector<unsigned> > wsv(NNODES);
#ifdef ONESYMMETRIC
	unsigned bound_i = NEDGES/2;
#else
	unsigned bound_i = NEDGES;
#endif
	for (unsigned i = 0; i < bound_i; ++i) {
		unsigned n1;
		unsigned n2;
		unsigned wt;
		fscanf(fin, "%u%u%u", &n1, &n2, &wt);
		//n1s[i] = n1;
		//n2s[i] = n2;
		//insert_sort(n1s, n2s, n1, n2, i);
#ifdef ONESYMMETRIC
		n1sv[n1-1].push_back(n2);
		n1sv[n2-1].push_back(n1);
#else
		n1sv[n1].push_back(n2);
		wsv[n1].push_back(wt);
#endif
		if (i % 10000000 == 0) {
			now = omp_get_wtime();
			printf("time: %lf, got %u 10M edges...\n", now - start, i/10000000);//test
		}
	}
	unsigned edge_id = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		for (unsigned j = 0; j < n1sv[i].size(); ++j) {
			n1s[edge_id] = i + 1;
			n2s[edge_id] = n1sv[i][j] + 1;
			weits[edge_id] = wsv[i][j];
			edge_id++;
		}
	}
	printf("Got origin data: %s\n", filename);

	string prefix = string(filename) + "_weighted";
	FILE *fout = fopen(prefix.c_str(), "w");
	fprintf(fout, "%u %u\n", NNODES, NEDGES);
	for (unsigned i = 0; i < NEDGES; ++i) {
		fprintf(fout, "%u %u %u\n", n1s[i], n2s[i], weits[i]);
	}
	// Clean the vectors for saving memory
	fclose(fin);
	fclose(fout);
	free(n1s);
	free(n2s);
	free(weits);
}

int main(int argc, char *argv[]) {
	start = omp_get_wtime();
	char *filename;
	if (argc > 3) {
		filename = argv[1];
		NNODES = strtoul(argv[2], NULL, 0);
		NEDGES = strtoul(argv[3], NULL, 0);
		//TILE_WIDTH = strtoul(argv[2], NULL, 0);
	} else {
		//filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		//TILE_WIDTH = 1024;
		printf("Usage: ./page_rank <data_file from Galois' converting> <num of vertices> <num of edges>\n");
		exit(1);
	}
	input_untiled(filename);
	return 0;
}
