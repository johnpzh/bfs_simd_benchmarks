#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <cstring>
#include <omp.h>
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
//using std::find_last_of;

#define DUMP 0.85
#define MAX_NODES 1700000
#define MAX_EDGES 40000000

struct Graph {
	int n1[MAX_EDGES];
	int n2[MAX_EDGES];
	int nneibor[MAX_NODES];
};

int nnodes, nedges;
Graph grah;
float rank[MAX_NODES];
float sum[MAX_NODES];
unsigned NUM_THREADS;

void input(char filename[]) {
	//printf("data: %s\n", filename);
	FILE *fin = fopen(filename, "r");

	fscanf(fin, "%u %u", &nnodes, &nedges);
	for (unsigned i = 0; i < nnodes; ++i) {
		grah.nneibor[i] = 0;
	}
	unsigned max_num_neibor = 0;
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		grah.n1[i] = n1;
		grah.n2[i] = n2;
		grah.nneibor[n1]++;
		if (grah.nneibor[n1] > max_num_neibor) {
			max_num_neibor = grah.nneibor[n1];
		}
	}
	fclose(fin);
	
	unsigned *degree_counters = (unsigned *) malloc((max_num_neibor + 1) * sizeof(unsigned));
	memset(degree_counters, 0, (max_num_neibor + 1)*sizeof(unsigned));
	for (unsigned i = 0; i < nnodes; ++i) {
		degree_counters[grah.nneibor[i]]++;
	}
	unsigned loc = string(filename).find_last_of("/") + 1;
	string data_file = string(filename).substr(loc);
	data_file = "degree_" + data_file;
	FILE *fout = fopen(data_file.c_str(), "w");
	for (unsigned i = 0; i < max_num_neibor + 1; ++i) {
		fprintf(fout, "%u %u\n", i, degree_counters[i]);
	}
	free(degree_counters);
}


int main(int argc, char *argv[]) {
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		NUM_THREADS = strtoul(argv[2], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/soc-pokec-relationships.txt";
		NUM_THREADS = 256;
	}
	input(filename);
	return 0;
}
