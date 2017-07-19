#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <omp.h>
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;

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
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		grah.n1[i] = n1;
		grah.n2[i] = n2;
		grah.nneibor[n1]++;
	}
	fclose(fin);
}

void input2(string filename, int tilesize) {
	ifstream fin(filename.c_str());
	string line;
	getline(fin, line);
	stringstream sin(line);
	sin >> nnodes >> nedges;

	for(int i=0;i<nnodes;i++) {
		grah.nneibor[i] = 0;
	}

	int cur = 0;
	while(getline(fin, line)) {
		int n, n1, n2;
		stringstream sin1(line);
		while(sin1 >> n) {
			grah.n1[cur] = n / tilesize;
			grah.n2[cur] = n % tilesize;
			cur++;
		}
	}
	nedges = cur;
}

void page_rank() {
#pragma omp parallel for num_threads(256)
	for(unsigned i=0;i<nnodes;i++) {
		rank[i] = 1.0;
		sum[i] = 0.0;
	}

	//for(int i=0;i<10;i++) {
	double start_time = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THREADS)
	for(unsigned j=0;j<nedges;j++) {
		int n1 = grah.n1[j];
		int n2 = grah.n2[j];
#pragma omp atomic
		sum[n2] += rank[n1]/grah.nneibor[n1];
	}
	//cout << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0 << endl;
	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);

	for(unsigned j = 0; j < nnodes; j++) {
		rank[j] = (1 - DUMP) / nnodes + DUMP * sum[j]; 	
	}
	//}
}

void print() {
	FILE *fout = fopen("ranks.txt", "w");
	for(unsigned i=0;i<nnodes;i++) {
		//cout << rank[i] << " ";
		fprintf(fout, "%lf\n", rank[i]);
	}
	//cout << endl;
	fclose(fout);
}

int main(int argc, char *argv[]) {
	double input_start = omp_get_wtime();
	//if(argc==2)
	//	input3(filename);
	//else
	//	input2(filename, 1024);
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		NUM_THREADS = strtoul(argv[2], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/soc-pokec-relationships.txt";
		NUM_THREADS = 256;
	}
	input(filename);
	double input_end = omp_get_wtime();
	//printf("input tims: %lf\n", input_end - input_start);
	page_rank();
	print();
	return 0;
}
