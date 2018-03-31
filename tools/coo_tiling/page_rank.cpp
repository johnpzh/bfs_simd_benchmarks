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

void input_weighted(char filename[], unsigned min_tile_width, unsigned max_tile_width) {
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	string prefix = string(filename) + "_untiled";
	//string prefix = string(filename) + "_reordered";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	fscanf(fin, "%u %u", &nnodes, &nedges);
	fclose(fin);
	
	// Read data (sorted)
	unsigned *n1s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *weights = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned NUM_THREADS = 64;
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
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = nedges;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		unsigned n1;
		unsigned n2;
		unsigned wt;
		fscanf(fin, "%u%u%u", &n1, &n2, &wt);
		n1s[index] = n1;
		n2s[index] = n2;
		weights[index] = wt;
	}
	//if (NUM_THREADS - 1 != tid) {
	//	for (unsigned i = 0; i < edge_bound; ++i) {
	//		unsigned index = i + offset;
	//		unsigned n1;
	//		unsigned n2;
	//		fscanf(fin, "%u %u", &n1, &n2);
	//		n1--;
	//		n2--;
	//		n1s[index] = n1;
	//		n2s[index] = n2;
	//	}
	//} else {
	//	for (unsigned i = 0; i + offset < nedges; ++i) {
	//		unsigned index = i + offset;
	//		unsigned n1;
	//		unsigned n2;
	//		fscanf(fin, "%u %u", &n1, &n2);
	//		n1--;
	//		n2--;
	//		n1s[index] = n1;
	//		n2s[index] = n2;
	//	}
	//}
	fclose(fin);
}
	printf("Got origin data: %s\n", filename);

	////////////////////////////////////////////////////////////	
	// Multi-version output
	for (TILE_WIDTH = min_tile_width; TILE_WIDTH <= max_tile_width; TILE_WIDTH *= 2) {

	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	memset(nneibor, 0, nnodes * sizeof(unsigned));
	unsigned num_tiles;
	unsigned side_length;
	if (nnodes % TILE_WIDTH) {
		side_length = nnodes / TILE_WIDTH + 1;
	} else {
		side_length = nnodes / TILE_WIDTH;
	}
	num_tiles = side_length * side_length;
	//if (nedges/num_tiles < 1) {
	//	printf("nedges: %u, num_tiles: %u, average: %u\n", nedges, num_tiles, nedges/num_tiles);
	//	fprintf(stderr, "Error: the tile width %u is too small.\n", TILE_WIDTH);
	//	exit(2);
	//}
	vector< vector<unsigned> > tiles_n1v;
	tiles_n1v.resize(num_tiles);
	vector< vector<unsigned> > tiles_n2v;
	tiles_n2v.resize(num_tiles);
	vector< vector<unsigned> > tiles_wtv(num_tiles);
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1;
		unsigned n2;
		//fscanf(fin, "%u %u", &n1, &n2);
		n1 = n1s[i];
		n2 = n2s[i];
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
		tiles_wtv[tile_id].push_back(weights[i]);
	}
	unsigned *tiles_n1 = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *tiles_n2 = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *tiles_weights = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned edge_i = 0;
	for (unsigned i = 0; i < num_tiles; ++i) {
		unsigned bound = tiles_n1v[i].size();
		for (unsigned j = 0; j < bound; ++j) {
			tiles_n1[edge_i] = tiles_n1v[i][j];
			tiles_n2[edge_i] = tiles_n2v[i][j];
			tiles_weights[edge_i] = tiles_wtv[i][j];
			++edge_i;
		}
	}
	printf("Got tile data. Start writing...\n");
	

	// Write to files
	prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
	edge_bound = nedges / ALIGNED_BYTES;
	NUM_THREADS = 64;
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
		fprintf(fout, "%u %u %u\n", tiles_n1[index], tiles_n2[index], tiles_weights[index]);
	}
	//if (NUM_THREADS - 1 != tid) {
	//	for (unsigned i = 0; i < edge_bound; ++i) {
	//		unsigned index = i + offset;
	//		fprintf(fout, "%u %u\n", tiles_n1[index], tiles_n2[index]);
	//	}
	//} else {
	//	for (unsigned i = 0; i + offset < nedges; ++i) {
	//		unsigned index = i + offset;
	//		fprintf(fout, "%u %u\n", tiles_n1[index], tiles_n2[index]);
	//	}
	//}
	fclose(fout);
}
	printf("Main files done...\n");
	// Write offsets
	fname = prefix + "-offsets";
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
	printf("Done.\n");
	fclose(fout);
	free(nneibor);
	free(tiles_n1);
	free(tiles_n2);
	free(tiles_weights);
	}
	// ENd Multi-version output
	////////////////////////////////////////////////////////////

	// Clean the vectors for saving memory
	//fclose(fin);
	free(weights);
	free(n1s);
	free(n2s);
}


void input_data(char filename[], unsigned min_tile_width, unsigned max_tile_width) {
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	string prefix = string(filename) + "_untiled";
	//string prefix = string(filename) + "_reordered";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (NULL == fin) {
		printf("Error: cannot open file %s.\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%u %u", &nnodes, &nedges);
	fclose(fin);
	
	// Read data (sorted)
	unsigned *n1s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *n2s = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned NUM_THREADS = 64;
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
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = nedges;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u%u", &n1, &n2);
		n1s[index] = n1;
		n2s[index] = n2;
	}
	//if (NUM_THREADS - 1 != tid) {
	//	for (unsigned i = 0; i < edge_bound; ++i) {
	//		unsigned index = i + offset;
	//		unsigned n1;
	//		unsigned n2;
	//		fscanf(fin, "%u %u", &n1, &n2);
	//		n1--;
	//		n2--;
	//		n1s[index] = n1;
	//		n2s[index] = n2;
	//	}
	//} else {
	//	for (unsigned i = 0; i + offset < nedges; ++i) {
	//		unsigned index = i + offset;
	//		unsigned n1;
	//		unsigned n2;
	//		fscanf(fin, "%u %u", &n1, &n2);
	//		n1--;
	//		n2--;
	//		n1s[index] = n1;
	//		n2s[index] = n2;
	//	}
	//}
	fclose(fin);
}
	printf("Got origin data: %s\n", filename);

	////////////////////////////////////////////////////////////	
	// Multi-version output
	for (TILE_WIDTH = min_tile_width; TILE_WIDTH <= max_tile_width; TILE_WIDTH *= 2) {

	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	memset(nneibor, 0, nnodes * sizeof(unsigned));
	unsigned num_tiles;
	unsigned side_length;
	if (nnodes % TILE_WIDTH) {
		side_length = nnodes / TILE_WIDTH + 1;
	} else {
		side_length = nnodes / TILE_WIDTH;
	}
	num_tiles = side_length * side_length;
	//if (nedges/num_tiles < 1) {
	//	printf("nedges: %u, num_tiles: %u, average: %u\n", nedges, num_tiles, nedges/num_tiles);
	//	fprintf(stderr, "Error: the tile width %u is too small.\n", TILE_WIDTH);
	//	exit(2);
	//}
	vector< vector<unsigned> > tiles_n1v;
	tiles_n1v.resize(num_tiles);
	vector< vector<unsigned> > tiles_n2v;
	tiles_n2v.resize(num_tiles);
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1;
		unsigned n2;
		//fscanf(fin, "%u %u", &n1, &n2);
		n1 = n1s[i];
		n2 = n2s[i];
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
	unsigned edge_i = 0;
	for (unsigned i = 0; i < num_tiles; ++i) {
		unsigned bound = tiles_n1v[i].size();
		for (unsigned j = 0; j < bound; ++j) {
			tiles_n1[edge_i] = tiles_n1v[i][j];
			tiles_n2[edge_i] = tiles_n2v[i][j];
			++edge_i;
		}
	}
	printf("Got tile data. Start writing...\n");
	

	// Write to files
	prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
	edge_bound = nedges / ALIGNED_BYTES;
	NUM_THREADS = 64;
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
		fprintf(fout, "%u %u\n", tiles_n1[index], tiles_n2[index]);
	}
	//if (NUM_THREADS - 1 != tid) {
	//	for (unsigned i = 0; i < edge_bound; ++i) {
	//		unsigned index = i + offset;
	//		fprintf(fout, "%u %u\n", tiles_n1[index], tiles_n2[index]);
	//	}
	//} else {
	//	for (unsigned i = 0; i + offset < nedges; ++i) {
	//		unsigned index = i + offset;
	//		fprintf(fout, "%u %u\n", tiles_n1[index], tiles_n2[index]);
	//	}
	//}
	fclose(fout);
}
	printf("Main files done...\n");
	// Write offsets
	fname = prefix + "-offsets";
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
	printf("Done.\n");
	// Clean the vectors for saving memory
	//fclose(fin);
	fclose(fout);
	free(nneibor);
	free(tiles_n1);
	free(tiles_n2);
	}
	// ENd Multi-version output
	////////////////////////////////////////////////////////////
	free(n1s);
	free(n2s);

}

int main(int argc, char *argv[]) {
	char *filename;
	unsigned min_tile_width;
	unsigned max_tile_width;
	if (argc > 3) {
		filename = argv[1];
		//TILE_WIDTH = strtoul(argv[2], NULL, 0);
		min_tile_width = strtoul(argv[2], NULL, 0);
		max_tile_width = strtoul(argv[3], NULL, 0);
	} else {
		//filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		//TILE_WIDTH = 1024;
		printf("Usage: ./page_rank <data_file> <min_tile_width> <max_tile_width>\n");
		exit(1);
	}
#ifdef WEIGHTED
	input_weighted(filename, min_tile_width, max_tile_width);
#else
	input_data(filename, min_tile_width, max_tile_width);
#endif
	return 0;
}
