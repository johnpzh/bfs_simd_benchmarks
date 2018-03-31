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

void input_data(char filename[], unsigned min_tile_width, unsigned max_tile_width) {
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	string prefix = string(filename) + "_untiled_reverse";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
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
		fscanf(fin, "%u %u", &n1, &n2);
		n1s[index] = n1;
		n2s[index] = n2;
	}
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
	if (nedges/num_tiles < 1) {
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
	prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH) + "_reverse";
	NUM_THREADS = 64;
	edge_bound = nedges / NUM_THREADS;
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
	input_data(filename, min_tile_width, max_tile_width);
	return 0;
}
