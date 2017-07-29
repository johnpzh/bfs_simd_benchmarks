#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
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
unsigned tile_width;

void page_rank(unsigned **tiles_n1, unsigned **tiles_n2, unsigned *tops, unsigned num_tiles);

//void input(char filename[], unsigned tile_width) {
//	//printf("data: %s\n", filename);
//	FILE *fin = fopen(filename, "r");
//
//	fscanf(fin, "%u %u", &nnodes, &nedges);
//	memset(grah.nneibor, 0, sizeof(grah.nneibor));
//	//for (unsigned i = 0; i < nnodes; ++i) {
//	//	grah.nneibor[i] = 0;
//	//}
//	unsigned num_tiles;
//	if (nnodes % tile_width) {
//		num_tiles = nnodes / tile_width + 1;
//	} else {
//		num_tiles = nnodes / tile_width;
//	}
//	unsigned max_top = nedges / num_tiles * 16;
//	unsigned **tiles_n1 = (unsigned **) malloc(num_tiles * sizeof(unsigned *));
//	unsigned **tiles_n2 = (unsigned **) malloc(num_tiles * sizeof(unsigned *));
//	for (unsigned i = 0; i < num_tiles; ++i) {
//		tiles_n1[i] = (unsigned *) malloc(max_top * sizeof(unsigned));
//		tiles_n2[i] = (unsigned *) malloc(max_top * sizeof(unsigned));
//	}
//	unsigned *tops = (unsigned *) malloc(num_tiles * sizeof(unsigned));
//	memset(tops, 0, num_tiles * sizeof(unsigned));
//	for (unsigned i = 0; i < nedges; ++i) {
//		unsigned n1;
//		unsigned n2;
//		fscanf(fin, "%u %u", &n1, &n2);
//		//grah.n1[i] = n1;
//		//grah.n2[i] = n2;
//		unsigned tile_id = n2 % num_tiles;
//		//unsigned tile_id = n2 / tile_width;
//		unsigned *top = tops + tile_id;
//		tiles_n1[tile_id][*top] = n1;
//		tiles_n2[tile_id][*top] = n2;
//		(*top)++;
//		grah.nneibor[n1]++;
//	}
//	fclose(fin);
//
//	// PageRank
//	page_rank(tiles_n1, tiles_n2, tops, num_tiles);
//
//	// Free memory
//	for (unsigned i = 0; i < num_tiles; ++i) {
//		free(tiles_n1[i]);
//		free(tiles_n2[i]);
//	}
//	free(tiles_n1);
//	free(tiles_n2);
//	free(tops);
//}

void input(char filename[]) {
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", filename);
		exit(1);
	}

	fscanf(fin, "%u %u", &nnodes, &nedges);
	memset(grah.nneibor, 0, sizeof(grah.nneibor));
	unsigned long long num_tiles;
	unsigned side_length;
	if (nnodes % tile_width) {
		side_length = nnodes / tile_width + 1;
	} else {
		side_length = nnodes / tile_width;
	}
	num_tiles = side_length * side_length;
	//unsigned max_top = nedges / num_tiles * 16;
	unsigned max_top = tile_width * tile_width / 128;
	unsigned **tiles_n1 = (unsigned **) malloc(num_tiles * sizeof(unsigned *));
	unsigned **tiles_n2 = (unsigned **) malloc(num_tiles * sizeof(unsigned *));
	for (unsigned i = 0; i < num_tiles; ++i) {
		tiles_n1[i] = (unsigned *) malloc(max_top * sizeof(unsigned));
		tiles_n2[i] = (unsigned *) malloc(max_top * sizeof(unsigned));
	}
	unsigned *tops = (unsigned *) malloc(num_tiles * sizeof(unsigned));
	memset(tops, 0, num_tiles * sizeof(unsigned));
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		unsigned n1_id = n1 / tile_width;
		unsigned n2_id = n2 / tile_width;
		//unsigned n1_id = n1 % side_length;
		//unsigned n2_id = n2 % side_length;
		unsigned tile_id = n1_id * side_length + n2_id;
		unsigned *top = tops + tile_id;
		if (*top == max_top) {
			fprintf(stderr, "Error: the tile %u is full.\n", tile_id);
			exit(1);
		}
		tiles_n1[tile_id][*top] = n1;
		tiles_n2[tile_id][*top] = n2;
		(*top)++;
		grah.nneibor[n1]++;
	}
	fclose(fin);

	// PageRank
	for (unsigned i = 0; i < 9; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		page_rank(tiles_n1, tiles_n2, tops, num_tiles);
	}

	// Free memory
	for (unsigned i = 0; i < num_tiles; ++i) {
		free(tiles_n1[i]);
		free(tiles_n2[i]);
	}
	free(tiles_n1);
	free(tiles_n2);
	free(tops);
}

void page_rank(unsigned **tiles_n1, unsigned **tiles_n2, unsigned *tops, unsigned num_tiles) {
#pragma omp parallel for num_threads(256)
	for(unsigned i=0;i<nnodes;i++) {
		rank[i] = 1.0;
		sum[i] = 0.0;
	}

	//for(int i=0;i<10;i++) {
	double start_time = omp_get_wtime();

//#pragma omp parallel for num_threads(NUM_THREADS)
//	for(unsigned j=0;j<nedges;j++) {
//		int n1 = grah.n1[j];
//		int n2 = grah.n2[j];
//#pragma omp atomic
//		sum[n2] += rank[n1]/grah.nneibor[n1];
//	}

#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < num_tiles; ++i) {
		unsigned top = tops[i];
		for (unsigned j = 0; j < top; ++j) {
			unsigned n1 = tiles_n1[i][j];
			unsigned n2 = tiles_n2[i][j];
			sum[n2] += rank[n1]/grah.nneibor[n1];
		}
	}

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
	if (argc > 3) {
		filename = argv[1];
		NUM_THREADS = strtoul(argv[2], NULL, 0);
		tile_width = strtoul(argv[3], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/soc-pokec-relationships.txt";
		NUM_THREADS = 256;
		tile_width = 8192;
	}
	input(filename);
	double input_end = omp_get_wtime();
	//printf("input tims: %lf\n", input_end - input_start);
	//page_rank(tile_width);
#ifdef ONEDEBUG
	print();
#endif
	return 0;
}
