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
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
using std::vector;
using std::to_string;

#define DUMP 0.85

int nnodes, nedges;
unsigned NUM_THREADS;
unsigned TILE_WIDTH;
unsigned CHUNK_SIZE;

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *nneibor, unsigned *tops, float *rank, float *sum, unsigned *offsets, unsigned num_tiles);
void print(float *rank);


//#define MAX_NODES 1700000
//#define MAX_EDGES 40000000

/////////////////////////////////////////////////////////////////////////
//struct Graph {
//	int n1[MAX_EDGES];
//	int n2[MAX_EDGES];
//	int nneibor[MAX_NODES];
//};
//Graph grah;
//float rank[MAX_NODES];
//float sum[MAX_NODES];
//void page_rank();
//void input(char filename[]) {
//	//printf("data: %s\n", filename);
//	FILE *fin = fopen(filename, "r");
//
//	fscanf(fin, "%u %u", &nnodes, &nedges);
//	for (unsigned i = 0; i < nnodes; ++i) {
//		grah.nneibor[i] = 0;
//	}
//	for (unsigned i = 0; i < nedges; ++i) {
//		unsigned n1;
//		unsigned n2;
//		fscanf(fin, "%u %u", &n1, &n2);
//		n1--;
//		n2--;
//		grah.n1[i] = n1;
//		grah.n2[i] = n2;
//		grah.nneibor[n1]++;
//	}
//	fclose(fin);
//	// PageRank
//#ifdef ONEDEBUG
//	page_rank();
//#else
//	for (unsigned i = 0; i < 9; ++i) {
//		NUM_THREADS = (unsigned) pow(2, i);
//		page_rank();
//	}
//#endif
//}
//
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
////////////////////////////////////////////////////////////////////////////

void input(char filename[]) {
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	string prefix = string(filename) + "_tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-" + to_string(0);
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%u %u", &nnodes, &nedges);
	fclose(fin);
	unsigned num_tiles;
	unsigned side_length;
	if (nnodes % TILE_WIDTH) {
		side_length = nnodes / TILE_WIDTH + 1;
	} else {
		side_length = nnodes / TILE_WIDTH;
	}
	num_tiles = side_length * side_length;
	// Read the offset and number of edges for every tile
	fname = prefix + "-offsets";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	unsigned *offsets = (unsigned *) malloc(num_tiles * sizeof(unsigned));
	for (unsigned i = 0; i < num_tiles; ++i) {
		fscanf(fin, "%u", offsets + i);
	}
	fclose(fin);
	unsigned *tops = (unsigned *) malloc(num_tiles * sizeof(unsigned));
	//memset(tops, 0, num_tiles * sizeof(unsigned));
	for (unsigned i = 0; i < num_tiles; ++i) {
		if (i != num_tiles - 1) {
			tops[i] = offsets[i + 1] - offsets[i];
		} else {
			tops[i] = nedges - offsets[i];
		}
	}
	// Read nneibor
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	for (unsigned i = 0; i < nnodes; ++i) {
		fscanf(fin, "%u", nneibor + i);
	}
	fclose(fin);
	// Read tiles
	unsigned *tiles_n1 = (unsigned *) _mm_malloc(nedges * sizeof(unsigned), ALIGNED_BYTES);
	unsigned *tiles_n2 = (unsigned *) _mm_malloc(nedges * sizeof(unsigned), ALIGNED_BYTES);
	unsigned edge_bound = nedges / ALIGNED_BYTES;
	NUM_THREADS = 64;
#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * edge_bound;
	fname = prefix + "-" + to_string(tid);
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	if (0 == tid) {
		fscanf(fin, "%u %u", &nnodes, &nedges);
	}
	if (NUM_THREADS - 1 != tid) {
		for (unsigned i = 0; i < edge_bound; ++i) {
			unsigned index = i + offset;
			unsigned n1;
			unsigned n2;
			fscanf(fin, "%u %u", &n1, &n2);
			n1--;
			n2--;
			tiles_n1[index] = n1;
			tiles_n2[index] = n2;
			//fscanf(fin, "%u %u", tiles_n1 + index, tiles_n2 + index);
		}
	} else {
		for (unsigned i = 0; i + offset < nedges; ++i) {
			unsigned index = i + offset;
			unsigned n1;
			unsigned n2;
			fscanf(fin, "%u %u", &n1, &n2);
			n1--;
			n2--;
			tiles_n1[index] = n1;
			tiles_n2[index] = n2;
			//fscanf(fin, "%u %u", tiles_n1 + index, tiles_n2 + index);
		}
	}
	fclose(fin);
}


	float *rank = (float *) malloc(nnodes * sizeof(float));
	float *sum = (float *) malloc(nnodes * sizeof(float));
	// PageRank
	for (unsigned i = 0; i < 9; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		page_rank(tiles_n1, tiles_n2, nneibor, tops, rank, sum, offsets, num_tiles);
	}

#ifdef ONEDEBUG
	print(rank);
#endif
	// Free memory
	_mm_free(tiles_n1);
	_mm_free(tiles_n2);
	free(nneibor);
	free(offsets);
	free(tops);
	free(rank);
	free(sum);
}

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *nneibor, unsigned *tops, float *rank, float *sum, unsigned *offsets, unsigned num_tiles) {
//void page_rank() {
#pragma omp parallel for num_threads(256)
	for(unsigned i=0;i<nnodes;i++) {
		rank[i] = 1.0;
		sum[i] = 0.0;
	}

	//for(int i=0;i<10;i++) {
	double start_time = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THREADS)
	for(unsigned j=0;j<nedges;j++) {
		//int n1 = grah.n1[j];
		//int n2 = grah.n2[j];
		int n1 = tiles_n1[j];
		int n2 = tiles_n2[j];
#pragma omp atomic
		//sum[n2] += rank[n1]/grah.nneibor[n1];
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
	//page_rank();
#ifdef ONEDEBUG
	print();
#endif
	return 0;
}
