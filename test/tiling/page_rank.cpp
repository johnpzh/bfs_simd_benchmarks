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


void input(char filename[]) {
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", filename);
		exit(1);
	}

	fscanf(fin, "%u %u", &nnodes, &nedges);
	unsigned *nneibor = (unsigned *) _mm_malloc(nnodes * sizeof(unsigned), ALIGNED_BYTES);
	memset(nneibor, 0, nnodes * sizeof(unsigned));
	unsigned num_tiles;
	unsigned side_length;
	if (nnodes % TILE_WIDTH) {
		side_length = nnodes / TILE_WIDTH + 1;
	} else {
		side_length = nnodes / TILE_WIDTH;
	}
	num_tiles = side_length * side_length;
	if (nedges/num_tiles < NUM_P_INT/2) {
		printf("nedges: %u, num_tiles: %u, average: %u\n", nedges, num_tiles, nedges/num_tiles);
		fprintf(stderr, "Error: the tile width %u is too small.\n", TILE_WIDTH);
		exit(2);
	}
	vector< vector<unsigned> > tiles_n1v;
	tiles_n1v.resize(num_tiles);
	vector< vector<unsigned> > tiles_n2v;
	tiles_n2v.resize(num_tiles);
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		unsigned n1_id = n1 / TILE_WIDTH;
		unsigned n2_id = n2 / TILE_WIDTH;
		unsigned tile_id = n1_id * side_length + n2_id;
		nneibor[n1]++;
		n1++;
		n2++;
		tiles_n1v[tile_id].push_back(n1);
		tiles_n2v[tile_id].push_back(n2);
	}
	unsigned *tiles_n1 = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *tiles_n2 = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned tile_index = 0;
	for (unsigned i = 0; i < num_tiles; ++i) {
		unsigned bound = tiles_n1v[i].size();
		for (unsigned j = 0; j < bound; ++j) {
			tiles_n1[tile_index] = tiles_n1v[i][j];
			tiles_n2[tile_index] = tiles_n2v[i][j];
			++tile_index;
		}
	}

	string prefix = string(filename) + "_tiled-" + to_string(TILE_WIDTH);
	unsigned edge_bound = nedges / ALIGNED_BYTES;
	unsigned NUM_THREADS = 64;
#pragma omp parallel num_threads(NUM_THREADS)
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * edge_bound;
	string fname = prefix + "-" + to_string(tid);
	FILE *fout = fopen(fname.c_str(), "w");
	if (0 == tid) {
		fprintf(fout, "%u %u\n", nnodes, nedges);
	}
	if (NUM_THREADS - 1 != tid) {
		for (unsigned i = 0; i < edge_bound; ++i) {
			unsigned index = i + offset;
			fprintf(fout, "%u %u\n", tiles_n1[index], tiles_n2[index]);
		}
	} else {
		for (unsigned i = 0; i + offset < nedges; ++i) {
			unsigned index = i + offset;
			fprintf(fout, "%u %u\n", tiles_n1[index], tiles_n2[index]);
		}
	}
	fclose(fout);
}
	string fname = prefix + "-offsets";
	FILE *fout = fopen(fname.c_str(), "w");
	unsigned offset = 0;
	for (unsigned i = 0; i < num_tiles; ++i) {
		unsigned size = tiles_n1v[i].size();
		fprintf(fout, "%u\n", offset);//Format: offset
		offset += size;
	}
	fclose(fout);
	fname = prefix + "-nneibor";
	fout = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < nnodes; ++i) {
		fprintf(fout, "%u\n", nneibor[i]);
	}
	// Clean the vectors for saving memory
	fclose(fin);
	fclose(fout);
	_mm_free(nneibor);
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
	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	memset(nneibor, 0, nnodes * sizeof(unsigned));
	//unsigned num_tiles;
	//unsigned side_length;
	//if (nnodes % TILE_WIDTH) {
	//	side_length = nnodes / TILE_WIDTH + 1;
	//} else {
	//	side_length = nnodes / TILE_WIDTH;
	//}
	//num_tiles = side_length * side_length;
	//if (nedges/num_tiles < NUM_P_INT/2) {
	//	printf("nedges: %u, num_tiles: %u, average: %u\n", nedges, num_tiles, nedges/num_tiles);
	//	fprintf(stderr, "Error: the tile width %u is too small.\n", TILE_WIDTH);
	//	exit(2);
	//}
	//vector< vector<unsigned> > tiles_n1v;
	//tiles_n1v.resize(num_tiles);
	//vector< vector<unsigned> > tiles_n2v;
	//tiles_n2v.resize(num_tiles);
	unsigned *n1s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(nedges * sizeof(unsigned));
	for (unsigned i = 0; i < nedges; ++i) {
		//fscanf(fin, "%u %u", n1s + i, n2s + i);
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1s[i] = n1;
		n2s[i] = n2;
		n1--;
		n2--;
		nneibor[n1]++;
		//unsigned n1_id = n1 / TILE_WIDTH;
		//unsigned n2_id = n2 / TILE_WIDTH;
		//unsigned tile_id = n1_id * side_length + n2_id;
		//nneibor[n1]++;
		//n1++;
		//n2++;
		//tiles_n1v[tile_id].push_back(n1);
		//tiles_n2v[tile_id].push_back(n2);
	}
	//unsigned *tiles_n1 = (unsigned *) malloc(nedges * sizeof(unsigned));
	//unsigned *tiles_n2 = (unsigned *) malloc(nedges * sizeof(unsigned));
	//unsigned tile_index = 0;
	//for (unsigned i = 0; i < num_tiles; ++i) {
	//	unsigned bound = tiles_n1v[i].size();
	//	for (unsigned j = 0; j < bound; ++j) {
	//		tiles_n1[tile_index] = tiles_n1v[i][j];
	//		tiles_n2[tile_index] = tiles_n2v[i][j];
	//		++tile_index;
	//	}
	//}

	//string prefix = string(filename) + "_tiled-" + to_string(TILE_WIDTH);
	string prefix = string(filename) + "_untiled";
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
	if (NUM_THREADS - 1 != tid) {
		for (unsigned i = 0; i < edge_bound; ++i) {
			unsigned index = i + offset;
			fprintf(fout, "%u %u\n", n1s[index], n2s[index]);
		}
	} else {
		for (unsigned i = 0; i + offset < nedges; ++i) {
			unsigned index = i + offset;
			fprintf(fout, "%u %u\n", n1s[index], n2s[index]);
		}
	}
	fclose(fout);
}
	//string fname = prefix + "-offsets";
	//FILE *fout = fopen(fname.c_str(), "w");
	//unsigned offset = 0;
	//for (unsigned i = 0; i < num_tiles; ++i) {
	//	unsigned size = tiles_n1v[i].size();
	//	fprintf(fout, "%u\n", offset);//Format: offset
	//	offset += size;
	//}
	//fclose(fout);
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
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec-relationships.txt";
		TILE_WIDTH = 1024;
	}
#ifdef UNTILE
	input_untiled(filename);
#else
	input(filename);
#endif
	return 0;
}
