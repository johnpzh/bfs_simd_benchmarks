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

#define DUMP 0.85
//#define MAX_NODES 1700000
//#define MAX_EDGES 40000000
//
//struct Graph {
//	int n1[MAX_EDGES];
//	int n2[MAX_EDGES];
//	int nneibor[MAX_NODES];
//};
//Graph grah;
//float rank[MAX_NODES];
//float sum[MAX_NODES];

int nnodes, nedges;
unsigned NUM_THREADS;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *nneibor, float *rank, float *sum);
void print(float *rank);

void input(char filename[]) {
	//printf("data: %s\n", filename);
	FILE *fin = fopen(filename, "r");

	fscanf(fin, "%u %u", &nnodes, &nedges);
	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	memset(nneibor, 0, nnodes * sizeof(unsigned));
	//for (unsigned i = 0; i < nnodes; ++i) {
	//	grah.nneibor[i] = 0;
	//}
	unsigned *tiles_n1 = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *tiles_n2 = (unsigned *) malloc(nedges * sizeof(unsigned));
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		//grah.n1[i] = n1;
		//grah.n2[i] = n2;
		//grah.nneibor[n1]++;
		tiles_n1[i] = n1;
		tiles_n2[i] = n2;
		nneibor[n1]++;
	}
	fclose(fin);
	float *rank = (float *) malloc(nnodes * sizeof(float));
	float *sum = (float *) malloc(nnodes * sizeof(float));
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
	// PageRank
	for (unsigned i = 0; i < 9; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		sleep(10);
		page_rank(tiles_n1, tiles_n2, nneibor, rank, sum);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);
#ifdef ONEDEBUG
	print(rank);
#endif
	// Free memory
	free(nneibor);
	free(tiles_n1);
	free(tiles_n2);
	free(rank);
	free(sum);
}

//void input2(string filename, int tilesize) {
//	ifstream fin(filename.c_str());
//	string line;
//	getline(fin, line);
//	stringstream sin(line);
//	sin >> nnodes >> nedges;
//
//	for(int i=0;i<nnodes;i++) {
//		grah.nneibor[i] = 0;
//	}
//
//	int cur = 0;
//	while(getline(fin, line)) {
//		int n, n1, n2;
//		stringstream sin1(line);
//		while(sin1 >> n) {
//			grah.n1[cur] = n / tilesize;
//			grah.n2[cur] = n % tilesize;
//			cur++;
//		}
//	}
//	nedges = cur;
//}

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *nneibor, float *rank, float *sum) {
#pragma omp parallel for num_threads(256)
	for(unsigned i=0;i<nnodes;i++) {
		rank[i] = 1.0;
		sum[i] = 0.0;
	}

	//for(int i=0;i<10;i++) {
	double start_time = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THREADS)
	for(unsigned j=0;j<nedges;j++) {
		int n1 = tiles_n1[j];
		int n2 = tiles_n2[j];
#pragma omp atomic
		sum[n2] += rank[n1]/nneibor[n1];
	}
	//cout << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0 << endl;
	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);

	for(unsigned j = 0; j < nnodes; j++) {
		rank[j] = (1 - DUMP) / nnodes + DUMP * sum[j]; 	
	}
	//}
}

void print(float *rank) {
	FILE *fout = fopen("ranks.txt", "w");
	for(unsigned i=0;i<nnodes;i++) {
		//cout << rank[i] << " ";
		fprintf(fout, "%lf\n", rank[i]);
	}
	//cout << endl;
	fclose(fout);
}

int main(int argc, char *argv[]) {
	start = omp_get_wtime();
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		NUM_THREADS = strtoul(argv[2], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec-relationships.txt";
		NUM_THREADS = 256;
	}
	input(filename);
	return 0;
}
