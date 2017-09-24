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
#include <igraph.h>
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
using std::vector;
using std::to_string;

#define DUMP 0.85

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

unsigned nnodes, nedges;
//unsigned TILE_WIDTH;

double start;
double now;

void input(char filename[]) {
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	string prefix = string(filename) + "_untiled";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	fscanf(fin, "%u %u", &nnodes, &nedges);
	fclose(fin);
	//nnodes = 10;
	//nedges = 9;
	
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
	//// TEST
	//string prefix;
	//string fname;
	//FILE *fin;
	//unsigned ijji = 0;
	//n1s[ijji] = 2; n2s[ijji++] = 6;
	//n1s[ijji] = 2; n2s[ijji++] = 7;
	//n1s[ijji] = 2; n2s[ijji++] = 3;
	//n1s[ijji] = 1; n2s[ijji++] = 8;
	//n1s[ijji] = 1; n2s[ijji++] = 9;
	//n1s[ijji] = 1; n2s[ijji++] = 2;
	//n1s[ijji] = 1; n2s[ijji++] = 10;
	//n1s[ijji] = 3; n2s[ijji++] = 4;
	//n1s[ijji] = 3; n2s[ijji++] = 5;
	////
	//// END TEST
	printf("Got origin data: %s\n", filename);


	// Community Detection
	igraph_vector_t edges_vec;
	igraph_vector_init(&edges_vec, 2 * nedges);
	unsigned e_i = 0;
	for (unsigned i = 0; i < nedges; ++i) {
		VECTOR(edges_vec)[e_i++] = n1s[i] - 1;
		VECTOR(edges_vec)[e_i++] = n2s[i] - 1;
	}
	printf("Preparing for detecting...\n");
	igraph_t agraph;
	igraph_create(&agraph, &edges_vec, nnodes, IGRAPH_UNDIRECTED);
	igraph_vector_destroy(&edges_vec);
	printf("Got a graph...\n");
	igraph_vector_t membership;
	igraph_vector_init(&membership, 0);
	printf("Detecting starts...\n");
	igraph_community_multilevel(&agraph, NULL, &membership, NULL, NULL);
	igraph_destroy(&agraph);
	printf("Detecting finished...\n");
#ifdef ONEDEBUG
	printf("membership length: %ld, nnodes: %u\n", igraph_vector_size(&membership), nnodes);
	//printf("Test membership: ");
	//for (int i = 0; i < igraph_vector_size(&membership); ++i) {
	//	printf("%d ", (int) VECTOR(membership)[i]);
	//}
	//printf("\n");
#endif

	// Getting the vertex ID map
	vector<unsigned> map_vertices(nnodes);
	vector<unsigned> is_mapped(nnodes, 0);
	unsigned map_id_value = 0;
	unsigned start_loc = 0;
	while (start_loc < nnodes) {
		if (is_mapped[start_loc]) {
			++start_loc;
			continue;
		}
		unsigned community_id = (unsigned) VECTOR(membership)[start_loc];
		for (unsigned i = start_loc; i < nnodes; ++i) {
			if ((unsigned) VECTOR(membership)[i] != community_id || \
					is_mapped[i]) {
				continue;
			}
			is_mapped[i] = 1;
			map_vertices[i] = map_id_value++;
		}
		++start_loc;
	}
	printf("Got the map.\n");

	// Reorder the graph
	vector< vector<unsigned> > n1sv(nnodes);
	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	memset(nneibor, 0, nnodes * sizeof(unsigned));
//	NUM_THREADS = 64;
//#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1 = n1s[i];
		unsigned n2 = n2s[i];
		--n1;
		--n2;
		unsigned n1r = map_vertices[n1];
		unsigned n2r = map_vertices[n2];
//#pragma omp atomic
		nneibor[n1r]++;
		//++n1r;
		//++n2r;
		//n1s[i] = n1r;
		//n2s[i] = n2r;
		n1sv[n1r].push_back(n2r);
	}
	unsigned edge_id = 0;
	for (unsigned i = 0; i < nnodes; ++i) {
		for (unsigned j = 0; j < n1sv[i].size(); ++j) {
			n1s[edge_id] = i + 1;
			n2s[edge_id] = n1sv[i][j] + 1;
			edge_id++;
		}
	}

	//unsigned num_tiles;
	//unsigned side_length;
	//if (nnodes % TILE_WIDTH) {
	//	side_length = nnodes / TILE_WIDTH + 1;
	//} else {
	//	side_length = nnodes / TILE_WIDTH;
	//}
	//num_tiles = side_length * side_length;
	//if (nedges/num_tiles < 1) {
	//	printf("nedges: %u, num_tiles: %u, average: %u\n", nedges, num_tiles, nedges/num_tiles);
	//	fprintf(stderr, "Error: the tile width %u is too small.\n", TILE_WIDTH);
	//	exit(2);
	//}
	//vector< vector<unsigned> > tiles_n1v;
	//tiles_n1v.resize(num_tiles);
	//vector< vector<unsigned> > tiles_n2v;
	//tiles_n2v.resize(num_tiles);
	//for (unsigned i = 0; i < nedges; ++i) {
	//	unsigned n1;
	//	unsigned n2;
	//	//fscanf(fin, "%u %u", &n1, &n2);
	//	n1 = n1s[i];
	//	n2 = n2s[i];
	//	n1--;
	//	n2--;
	//	unsigned n1_id = n1 / TILE_WIDTH;
	//	unsigned n2_id = n2 / TILE_WIDTH;
	//	unsigned tile_id = n1_id * side_length + n2_id;
	//	nneibor[n1]++;
	//	n1++;
	//	n2++;
	//	tiles_n1v[tile_id].push_back(n1);
	//	tiles_n2v[tile_id].push_back(n2);
	//}
	//free(n1s);
	//free(n2s);
	//unsigned *tiles_n1 = (unsigned *) malloc(nedges * sizeof(unsigned));
	//unsigned *tiles_n2 = (unsigned *) malloc(nedges * sizeof(unsigned));
	//unsigned edge_i = 0;
	//for (unsigned i = 0; i < num_tiles; ++i) {
	//	unsigned bound = tiles_n1v[i].size();
	//	for (unsigned j = 0; j < bound; ++j) {
	//		tiles_n1[edge_i] = tiles_n1v[i][j];
	//		tiles_n2[edge_i] = tiles_n2v[i][j];
	//		++edge_i;
	//	}
	//}
	printf("Got tile data. Start writing...\n");
	

	// Write to files
	//prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
	prefix = string(filename) + "_reordered";
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
		//fprintf(fout, "%u %u\n", tiles_n1[index], tiles_n2[index]);
		fprintf(fout, "%u %u\n", n1s[index], n2s[index]);
	}
	fclose(fout);
}
	printf("Main files done...\n");
	// Write offsets
	//fname = prefix + "-offsets";
	//FILE *fout = fopen(fname.c_str(), "w");
	//unsigned offset = 0;
	//for (unsigned i = 0; i < num_tiles; ++i) {
	//	unsigned size = tiles_n1v[i].size();
	//	fprintf(fout, "%u\n", offset);//Format: offset
	//	offset += size;
	//}
	//fclose(fout);
	fname = prefix + "-nneibor";
	FILE *fout = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < nnodes; ++i) {
		fprintf(fout, "%u\n", nneibor[i]);
	}
	printf("Done.\n");
	// Clean the vectors for saving memory
	//fclose(fin);
	free(n1s);
	free(n2s);
	fclose(fout);
	free(nneibor);
	igraph_vector_destroy(&membership);
}

int main(int argc, char *argv[]) {
	char *filename;
	if (argc > 1) {
		filename = argv[1];
	} else {
		//filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		filename = "data/data";
	}
	input(filename);
	return 0;
}
