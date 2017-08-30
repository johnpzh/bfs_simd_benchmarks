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
using std::to_string;

#define DUMP 0.85
#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

unsigned nnodes, nedges;
unsigned NUM_THREADS;
unsigned TILE_WIDTH;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

//void page_rank(unsigned *n1s, unsigned *n2s, unsigned *nneibor, float *rank, float *sum);
void print(float *rank);

/////////////////////////////////////////////////////////////////////
//// Comment for save
/////////////////////////////////////////////////////////////////////
//void input(char filename[]) {
//	//printf("data: %s\n", filename);
//	string prefix = string(filename) + "_untiled";
//	string fname = prefix + "-0";
//	FILE *fin = fopen(fname.c_str(), "r");
//	fscanf(fin, "%u %u", &nnodes, &nedges);
//	fclose(fin);
//	unsigned *n1s = (unsigned *) malloc(nedges * sizeof(unsigned));
//	unsigned *n2s = (unsigned *) malloc(nedges * sizeof(unsigned));
//	NUM_THREADS = 64;
//	unsigned edge_bound = nedges / NUM_THREADS;
//#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
//{
//	unsigned tid = omp_get_thread_num();
//	unsigned offset = tid * edge_bound;
//	fname = prefix + "-" + to_string(tid);
//	fin = fopen(fname.c_str(), "r");
//	if (!fin) {
//		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
//		exit(1);
//	}
//	if (0 == tid) {
//		fscanf(fin, "%u %u\n", &nnodes, &nedges);
//	}
//	if (NUM_THREADS - 1 != tid) {
//		for (unsigned i = 0; i < edge_bound; ++i) {
//			unsigned index = i + offset;
//			unsigned n1;
//			unsigned n2;
//			fscanf(fin, "%u %u", &n1, &n2);
//			n1--;
//			n2--;
//			n1s[index] = n1;
//			n2s[index] = n2;
//		}
//	} else {
//		for (unsigned i = 0; i + offset < nedges; ++i) {
//			unsigned index = i + offset;
//			unsigned n1;
//			unsigned n2;
//			fscanf(fin, "%u %u", &n1, &n2);
//			n1--;
//			n2--;
//			n1s[index] = n1;
//			n2s[index] = n2;
//		}
//	}
//	fclose(fin);
//}
//	// Read nneibor
//	fname = prefix + "-nneibor";
//	fin = fopen(fname.c_str(), "r");
//	if (!fin) {
//		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
//		exit(1);
//	}
//	unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
//	for (unsigned i = 0; i < nnodes; ++i) {
//		fscanf(fin, "%u", nneibor + i);
//	}
//
//	float *rank = (float *) malloc(nnodes * sizeof(float));
//	float *sum = (float *) malloc(nnodes * sizeof(float));
//	now = omp_get_wtime();
//	time_out = fopen(time_file, "w");
//	fprintf(time_out, "input end: %lf\n", now - start);
//	// PageRank
//	for (unsigned i = 0; i < 9; ++i) {
//		NUM_THREADS = (unsigned) pow(2, i);
//		sleep(10);
//		page_rank(n1s, n2s, nneibor, rank, sum);
//		now = omp_get_wtime();
//		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
//	}
//	fclose(time_out);
//#ifdef ONEDEBUG
//	print(rank);
//#endif
//	// Free memory
//	free(nneibor);
//	free(n1s);
//	free(n2s);
//	free(rank);
//	free(sum);
//}

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *nneibor, unsigned *tops, float *rank, float *sum, unsigned *offsets, unsigned num_tiles, unsigned side_length, int *not_empty_tile);

void input(char filename[])
{
#ifdef ONEDEBUG
	printf("input: %s\n", filename);
#endif
	//string prefix = string(filename) + "_tiled-" + to_string(TILE_WIDTH);
	string prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
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
	int *not_empty_tile = (int *) malloc(num_tiles * sizeof(int));
	memset(not_empty_tile, 0, num_tiles * sizeof(int));
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
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = edge_bound + offset;
	} else {
		bound_index = nedges;
	}
	for (unsigned index = 0; index < bound_index; ++index) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		tiles_n1[index] = n1;
		tiles_n2[index] = n2;
		unsigned n1_id = n1 / TILE_WIDTH;
		unsigned n2_id = n2 / TILE_WIDTH;
		unsigned tile_id = n1_id * side_length + n2_id;
		not_empty_tile[tile_id] = 1;
	}

	//if (NUM_THREADS - 1 != tid) {
	//	for (unsigned i = 0; i < edge_bound; ++i) {
	//		unsigned index = i + offset;
	//		unsigned n1;
	//		unsigned n2;
	//		fscanf(fin, "%u %u", &n1, &n2);
	//		n1--;
	//		n2--;
	//		tiles_n1[index] = n1;
	//		tiles_n2[index] = n2;
	//		//fscanf(fin, "%u %u", tiles_n1 + index, tiles_n2 + index);
	//	}
	//} else {
	//	for (unsigned i = 0; i + offset < nedges; ++i) {
	//		unsigned index = i + offset;
	//		unsigned n1;
	//		unsigned n2;
	//		fscanf(fin, "%u %u", &n1, &n2);
	//		n1--;
	//		n2--;
	//		tiles_n1[index] = n1;
	//		tiles_n2[index] = n2;
	//		//fscanf(fin, "%u %u", tiles_n1 + index, tiles_n2 + index);
	//	}
	//}
	fclose(fin);
}

	float *rank = (float *) malloc(nnodes * sizeof(float));
	float *sum = (float *) malloc(nnodes * sizeof(float));
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned bound_i = 1;
#else
	unsigned bound_i = 9;
#endif
	// PageRank
	for (unsigned i = 0; i < bound_i; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#pragma omp parallel for num_threads(64)
		for (unsigned i = 0; i < nnodes; i++) {
			rank[i] = 1.0;
			sum[i] = 0.0;
		}
		sleep(10);
		page_rank(tiles_n1, tiles_n2, nneibor, tops, rank, sum, offsets, num_tiles, side_length, not_empty_tile);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	fclose(time_out);

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
	free(not_empty_tile);
}

/////////////////////
//void page_rank(unsigned *n1s, unsigned *n2s, unsigned *nneibor, float *rank, float *sum) {
//
//	//for(int i=0;i<10;i++) {
//	double start_time = omp_get_wtime();
//
//#pragma omp parallel for num_threads(NUM_THREADS)
//	for(unsigned j=0;j<nedges;j++) {
//		int n1 = n1s[j];
//		int n2 = n2s[j];
//#pragma omp atomic
//		sum[n2] += rank[n1]/nneibor[n1];
//	}
//	//cout << (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0 << endl;
//	double end_time = omp_get_wtime();
//	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
//
//	for(unsigned j = 0; j < nnodes; j++) {
//		rank[j] = (1 - DUMP) / nnodes + DUMP * sum[j]; 	
//	}
//	//}
//}
////////////////////////////

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *nneibor, unsigned *tops, float *rank, float *sum, unsigned *offsets, unsigned num_tiles, unsigned side_length, int *not_empty_tile)
{
	unsigned row_step = 1;
	unsigned row_index;
	double start_time = omp_get_wtime();
	for (row_index = 0; row_index <= side_length - row_step; row_index += row_step) {
#pragma omp parallel num_threads(NUM_THREADS)
	{
		unsigned tid = omp_get_thread_num();
		for (unsigned row_id = row_index; row_id < row_index + row_step; ++row_id) {
			for (unsigned col_id = tid; col_id < side_length; col_id += NUM_THREADS) {
				unsigned tile_id = row_id * side_length + col_id;
				if (!not_empty_tile[tile_id]) {
					continue;
				}
				unsigned bound_edge_i = offsets[tile_id] + tops[tile_id];
				for (unsigned edge_i = offsets[tile_id]; edge_i < bound_edge_i; ++edge_i) {
					unsigned n1 = tiles_n1[edge_i];
					unsigned n2 = tiles_n2[edge_i];
					sum[n2] += rank[n1]/nneibor[n1];
				}
			}
		}
	}
	}

#pragma omp parallel num_threads(NUM_THREADS)
	{
		unsigned tid = omp_get_thread_num();
		for (unsigned row_id = row_index; row_id < side_length; ++row_id) {
			for (unsigned col_id = tid; col_id < side_length; col_id += NUM_THREADS) {
				unsigned tile_id = row_id * side_length + col_id;
				if (!not_empty_tile[tile_id]) {
					continue;
				}
				unsigned bound_edge_i = offsets[tile_id] + tops[tile_id];
				for (unsigned edge_i = offsets[tile_id]; edge_i < bound_edge_i; ++edge_i) {
					unsigned n1 = tiles_n1[edge_i];
					unsigned n2 = tiles_n2[edge_i];
					sum[n2] += rank[n1]/nneibor[n1];
				}
			}
		}
	}

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);

#pragma omp parallel num_threads(64)
	for(unsigned j = 0; j < nnodes; j++) {
		rank[j] = (1 - DUMP) / nnodes + DUMP * sum[j]; 	
	}

}
void print(float *rank) 
{
	FILE *fout = fopen("ranks.txt", "w");
	for(unsigned i=0;i<nnodes;i++) {
		//cout << rank[i] << " ";
		fprintf(fout, "%lf\n", rank[i]);
	}
	//cout << endl;
	fclose(fout);
}

int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec-relationships.txt";
		TILE_WIDTH = 1024;
	}
	input(filename);
	return 0;
}
