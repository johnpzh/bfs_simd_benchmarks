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

#define DUMP 0.85
//#define MAX_NODES 1700000
//#define MAX_EDGES 40000000
#define MAX_NODES 67108864
#define MAX_EDGES 2147483648

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

//struct Graph {
//	//int n1[MAX_EDGES];
//	//int n2[MAX_EDGES];
//	int nneibor[MAX_NODES];
//};
//Graph grah;

unsigned nnodes, nedges;
//float rank[MAX_NODES];
//float sum[MAX_NODES];
unsigned TILE_WIDTH;

double start;
double now;


//void page_rank(unsigned *tops, unsigned *offsets, unsigned num_tiles);
//void print();

//////////////////////////////////////////////////////////////////////////////////
// Commented for clone
//void input(char filename[]) {
//	FILE *fin = fopen(filename, "r");
//	if (!fin) {
//		fprintf(stderr, "cannot open file: %s\n", filename);
//		exit(1);
//	}
//
//	fscanf(fin, "%u %u", &nnodes, &nedges);
//	memset(nneibor, 0, sizeof(nneibor));
//	unsigned num_tiles;
//	//unsigned long long num_tiles;
//	unsigned side_length;
//	if (nnodes % TILE_WIDTH) {
//		side_length = nnodes / TILE_WIDTH + 1;
//	} else {
//		side_length = nnodes / TILE_WIDTH;
//	}
//	num_tiles = side_length * side_length;
//	if (nedges < num_tiles) {
//		fprintf(stderr, "Error: tile size is too small.\n");
//		exit(2);
//	}
//	//unsigned max_top = nedges / num_tiles * 16;
//	unsigned max_top = TILE_WIDTH * TILE_WIDTH / 128;
//	unsigned **tiles_n1 = (unsigned **) _mm_malloc(num_tiles * sizeof(unsigned *), ALIGNED_BYTES);
//	unsigned **tiles_n2 = (unsigned **) _mm_malloc(num_tiles * sizeof(unsigned *), ALIGNED_BYTES);
//	for (unsigned i = 0; i < num_tiles; ++i) {
//		tiles_n1[i] = (unsigned *) _mm_malloc(max_top * sizeof(unsigned), ALIGNED_BYTES);
//		tiles_n2[i] = (unsigned *) _mm_malloc(max_top * sizeof(unsigned), ALIGNED_BYTES);
//	}
//	unsigned *tops = (unsigned *) _mm_malloc(num_tiles * sizeof(unsigned), ALIGNED_BYTES);
//	memset(tops, 0, num_tiles * sizeof(unsigned));
//	for (unsigned i = 0; i < nedges; ++i) {
//		unsigned n1;
//		unsigned n2;
//		fscanf(fin, "%u %u", &n1, &n2);
//		n1--;
//		n2--;
//		unsigned n1_id = n1 / TILE_WIDTH;
//		unsigned n2_id = n2 / TILE_WIDTH;
//		//unsigned n1_id = n1 % side_length;
//		//unsigned n2_id = n2 % side_length;
//		unsigned tile_id = n1_id * side_length + n2_id;
//
//		unsigned *top = tops + tile_id;
//		if (*top == max_top) {
//			fprintf(stderr, "Error: the tile %u is full.\n", tile_id);
//			exit(1);
//		}
//		tiles_n1[tile_id][*top] = n1;
//		tiles_n2[tile_id][*top] = n2;
//		(*top)++;
//		nneibor[n1]++;
//	}
//	fclose(fin);
//
//	// PageRank
//	for (unsigned i = 0; i < 9; ++i) {
//		NUM_THREADS = (unsigned) pow(2, i);
//		page_rank(tiles_n1, tiles_n2, tops, num_tiles);
//	}
//
//	// Free memory
//	for (unsigned i = 0; i < num_tiles; ++i) {
//		_mm_free(tiles_n1[i]);
//		_mm_free(tiles_n2[i]);
//	}
//	_mm_free(tiles_n1);
//	_mm_free(tiles_n2);
//	_mm_free(tops);
//}
////////////////////////////////////////////////////////////////////////////

void manual_sort(vector<unsigned> &n1v, vector<unsigned> &n2v)
{
	unsigned length = n1v.size();
	//for (unsigned i = length; i > 0; --i) {
	//	int swapped = 0;
	//	for (unsigned j = 0; j < i - 1; ++j) {
	//		if (n1v[j] > n1v[j+1]) {
	//			unsigned tmp = n1v[j];
	//			n1v[j] = n1v[j+1];
	//			n1v[j+1] = tmp;
	//			tmp = n2v[j];
	//			n2v[j] = n2v[j+1];
	//			n2v[j+1] = tmp;
	//			swapped = 1;
	//		}
	//	}
	//	if (!swapped) {
	//		break;
	//	}
	//}
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

void input_weighted(char filename[]) 
{
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", filename);
		exit(1);
	}

	fscanf(fin, "%u %u", &nnodes, &nedges);
#ifdef UNDIRECTED
	nedges *= 2;
#endif
	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	memset(nneibor, 0, nnodes * sizeof(unsigned));
	unsigned *n1s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *weights = (unsigned *) malloc(nedges * sizeof(unsigned));
	vector< vector<unsigned> > n1sv(nnodes);
	vector< vector<unsigned> > weights_v(nedges);//Weights
#ifdef UNDIRECTED
	unsigned bound_i = nedges/2;
#else
	unsigned bound_i = nedges;
#endif
	for (unsigned i = 0; i < bound_i; ++i) {
		unsigned n1;
		unsigned n2;
		unsigned wt;
		fscanf(fin, "%u%u%u", &n1, &n2, &wt);
		//n1s[i] = n1;
		//n2s[i] = n2;
		//insert_sort(n1s, n2s, n1, n2, i);
#ifdef UNDIRECTED
		n1sv[n1-1].push_back(n2);
		n1sv[n2-1].push_back(n1);
		weights_v[n1-1].push_back(wt);
		weights_v[n2-1].push_back(wt);
		nneibor[n1-1]++;
		nneibor[n2-1]++;
#else
		n1--;
		n1sv[n1].push_back(n2);
		weights_v[n1].push_back(wt);
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
			weights[edge_id] = weights_v[i][j];
			edge_id++;
		}
	}
	printf("Got origin data: %s\n", filename);

	string prefix = string(filename) + "_nohead";
	FILE *fout = fopen(prefix.c_str(), "w");
	//fprintf(fout, "%u %u\n", nnodes, nedges);
	for (unsigned i = 0; i < nedges; ++i) {
		fprintf(fout, "%u %u %u\n", n1s[i]-1, n2s[i]-1, weights[i]);
	}
	// Clean the vectors for saving memory
	fclose(fin);
	fclose(fout);
	free(nneibor);
	free(n1s);
	free(n2s);
	free(weights);
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
#ifdef UNDIRECTED
	nedges *= 2;
#endif
	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	memset(nneibor, 0, nnodes * sizeof(unsigned));
	unsigned *n1s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(nedges * sizeof(unsigned));
	vector< vector<unsigned> > n1sv(nnodes);
#ifdef UNDIRECTED
	unsigned bound_i = nedges/2;
#else
	unsigned bound_i = nedges;
#endif
	for (unsigned i = 0; i < bound_i; ++i) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		//n1s[i] = n1;
		//n2s[i] = n2;
		//insert_sort(n1s, n2s, n1, n2, i);
#ifdef UNDIRECTED
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

	string prefix = string(filename) + "_nohead";
	FILE *fout = fopen(prefix.c_str(), "w");
	//fprintf(fout, "%u %u\n", nnodes, nedges);
	for (unsigned i = 0; i < nedges; ++i) {
		fprintf(fout, "%u %u\n", n1s[i]-1, n2s[i]-1);
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
#ifdef WEIGHTED
	input_weighted(filename);
#else
	input_untiled(filename);
#endif
	//input(filename);
	return 0;
}
