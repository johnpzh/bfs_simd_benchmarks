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
unsigned CHUNK_SIZE;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

void page_rank(\
		unsigned *tiles_n1, \
		unsigned *tiles_n2, \
		unsigned *nneibor, \
		unsigned *tops, \
		float *rank, \
		float *sum, \
		unsigned *offsets, \
		unsigned num_tiles, \
		unsigned side_length, \
		unsigned *not_empty_tile, \
		unsigned row_step);
void print(float *rank);

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
	unsigned *not_empty_tile = (unsigned *) malloc(num_tiles * sizeof(unsigned));
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
	for (unsigned index = offset; index < bound_index; ++index) {
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
	fclose(fin);
}
	//unsigned *row_id_bounds = (unsigned *) malloc(sizeof(unsigned) * side_length);
	//unsigned *row_starts = (unsigned *) malloc(sizeof(unsigned) * side_length);
	//unsigned last_row_id = 0;
	//unsigned not_empty_count = 0;
	//row_starts[0] = 0;
	//for (unsigned tile_id = 0; tile_id < num_tiles; ++tile_id) {
	//	if (not_empty_tile[tile_id]) {
	//		not_empty_tile[not_empty_count] = tile_id;
	//		unsigned row_id = tile_id / side_length;
	//		if (last_row_id + 1 == row_id) {
	//			//row_id_bounds[last_row_id] = not_empty_count;
	//			row_starts[row_id] = not_empty_count;
	//			last_row_id = row_id;
	//		}
	//		++not_empty_count;
	//	}
	//}
	//++last_row_id;
	//while (last_row_id < side_length) {
	//	//row_id_bounds[last_row_id++] = not_empty_count;
	//	row_starts[last_row_id++] = not_empty_count;
	//}

	float *rank = (float *) malloc(nnodes * sizeof(float));
	float *sum = (float *) malloc(nnodes * sizeof(float));
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned bound_i = 2;
#else
	unsigned bound_i = 9;
#endif
	// PageRank
	for (unsigned row_step = 1; row_step < 10000; row_step *= 2) {
		printf("row_step: %u\n", row_step);//test
	//for (unsigned i = 0; i < 21; ++i) {
	//CHUNK_SIZE = (unsigned) pow(2, i);
	//printf("CHUNK_SIZE: %u\n", CHUNK_SIZE);//test
	CHUNK_SIZE = 1;
	//unsigned row_step = 128;
	//CHUNK_SIZE = 512;
	//unsigned row_step = 64;
	for (unsigned i = 0; i < bound_i; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#pragma omp parallel for num_threads(64)
		for (unsigned i = 0; i < nnodes; i++) {
			rank[i] = 1.0;
			sum[i] = 0.0;
		}
#ifndef ONEDEBUG
		//sleep(10);
#endif
		page_rank(\
				tiles_n1, \
				tiles_n2, \
				nneibor, \
				tops, \
				rank, \
				sum, \
				offsets, \
				num_tiles, \
				side_length, \
				not_empty_tile, \
				row_step);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
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

inline void scheduler(\
		unsigned row_index,\
		unsigned bound_row_id,\
		unsigned *not_empty_tile,\
		unsigned *tiles_n1,\
		unsigned *tiles_n2,\
		unsigned *offsets,\
		unsigned *tops,\
		float *sum,\
		float *rank,\
		unsigned *nneibor,\
		const unsigned &side_length)
{
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
	for (unsigned col_id = 0; col_id < side_length; ++col_id) {
		for (unsigned row_id = row_index; row_id < bound_row_id; ++row_id) {
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

void page_rank(\
		unsigned *tiles_n1, \
		unsigned *tiles_n2, \
		unsigned *nneibor, \
		unsigned *tops, \
		float *rank, \
		float *sum, \
		unsigned *offsets, \
		unsigned num_tiles, \
		unsigned side_length, \
		unsigned *not_empty_tile, \
		unsigned row_step)
{
	//unsigned row_step = 1;
	if (side_length < row_step) {
		printf("Error: row_step (%u) is to large, larger than side_length (%u)\n", \
				row_step, side_length);
		exit(3);
	}
	omp_set_num_threads(NUM_THREADS);
	unsigned row_index;
	double start_time = omp_get_wtime(); // Timer Starts
	for (row_index = 0; row_index <= side_length - row_step; row_index += row_step) {
		scheduler(\
				row_index,\
				row_index + row_step,\
				not_empty_tile,\
				tiles_n1,\
				tiles_n2,\
				offsets,\
				tops,\
				sum,\
				rank,\
				nneibor,\
				side_length);
	}

	if (row_step > 1) {
		scheduler(\
				row_index,\
				side_length,\
				not_empty_tile,\
				tiles_n1,\
				tiles_n2,\
				offsets,\
				tops,\
				sum,\
				rank,\
				nneibor,\
				side_length);
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
		filename = "/home/zpeng/benchmarks/data/pokec/coo_tiled_bak/soc-pokec";
		TILE_WIDTH = 1024;
	}
	input(filename);
	return 0;
}
