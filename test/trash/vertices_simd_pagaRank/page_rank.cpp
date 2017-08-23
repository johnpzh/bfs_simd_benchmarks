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
#include <unistd.h>
#include <hbwmalloc.h>
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;
using std::vector;
using std::to_string;

#define DUMP 0.85
//#define MAX_NODES 67108864
//#define MAX_EDGES 2147483648

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


void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *nneibor, unsigned *tops, float *rank, float *sum, unsigned *offsets, unsigned num_tiles);
void print(float *rank);

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
	//unsigned *nneibor = (unsigned *) malloc(nnodes * sizeof(unsigned));
	unsigned *nneibor = (unsigned *) hbw_malloc(nnodes * sizeof(unsigned));
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


	//float *rank = (float *) malloc(nnodes * sizeof(float));
	//float *sum = (float *) malloc(nnodes * sizeof(float));
	float *rank = (float *) hbw_malloc(nnodes * sizeof(float));
	float *sum = (float *) hbw_malloc(nnodes * sizeof(float));
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
	// PageRank
	for (unsigned i = 0; i < 9; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		sleep(10);
		page_rank(tiles_n1, tiles_n2, nneibor, tops, rank, sum, offsets, num_tiles);
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
	free(offsets);
	free(tops);
	//free(nneibor);
	//free(rank);
	//free(sum);
	hbw_free(nneibor);
	hbw_free(rank);
	hbw_free(sum);
}

inline void get_seq_sum(unsigned *n1s, unsigned *n2s, unsigned *nneibor, float *rank, float *sum, unsigned index, unsigned frontier)
{
	for (unsigned i = index; i < frontier; ++i) {
		unsigned n1 = n1s[i];
		unsigned n2 = n2s[i];
//#pragma omp atomic
		sum[n2] += rank[n1]/nneibor[n1];
	}
}

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *nneibor, unsigned *tops, float *rank, float *sum, unsigned *offsets, unsigned num_tiles) {
#ifdef ONEDEBUG
	printf("pageRanking...\n");
#endif
	const __m512i one_v = _mm512_set1_epi32(1);
	const __m512i zero_v = _mm512_set1_epi32(0);
	const __m512i minusone_v = _mm512_set1_epi32(-1);
#pragma omp parallel for num_threads(256)
	for(unsigned i=0;i<nnodes;i++) {
		rank[i] = 1.0;
		sum[i] = 0.0;
	}

	//for(int i=0;i<10;i++) {
	double start_time = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, CHUNK_SIZE)
	for (unsigned i = 0; i < num_tiles; ++i) {
		unsigned top = tops[i];
		unsigned j = 0;
		unsigned frontier;
		if (top <= NUM_P_INT) {
			frontier = 0;
		} else {
			frontier = top - NUM_P_INT;
		}
		for (; j < frontier; j += NUM_P_INT) {
			// Full loaded SIMD lanes
			__m512i n1_v = _mm512_load_epi32(tiles_n1 + offsets[i] + j);
			__m512i n2_v = _mm512_load_epi32(tiles_n2 + offsets[i] + j);
			__m512i conflict_n2 = _mm512_conflict_epi32(n2_v);
			__mmask16 is_conflict = _mm512_cmpneq_epi32_mask(conflict_n2, zero_v);
			if (*((short *)(&is_conflict)) == 0) {
				// No conflicts
				__m512 rank_v = _mm512_i32gather_ps(n1_v, rank, sizeof(float));
				__m512i nneibor_vi = _mm512_i32gather_epi32(n1_v, nneibor, sizeof(int));
				__m512 nneibor_v = _mm512_cvtepi32_ps(nneibor_vi);
				__m512 tmp_sum = _mm512_div_ps(rank_v, nneibor_v);
				__m512 sum_n2_v = _mm512_i32gather_ps(n2_v, sum, sizeof(float));
				tmp_sum = _mm512_add_ps(tmp_sum, sum_n2_v);
				_mm512_i32scatter_ps(sum, n2_v, tmp_sum, sizeof(float));
			} else {
				// Conflicts exists, then process sequentially
				get_seq_sum(tiles_n1 + offsets[i], tiles_n2 + offsets[i], nneibor, rank, sum, j, j + NUM_P_INT);
			}
		}
		// Process remain sequentially
		get_seq_sum(tiles_n1 + offsets[i], tiles_n2 + offsets[i], nneibor, rank, sum, j, top);
	}

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
	//printf("%u %lf\n", TILE_WIDTH, end_time - start_time);
	//printf("%u %lf\n", CHUNK_SIZE, end_time - start_time);

#pragma omp parallel for num_threads(256)
	for(unsigned j = 0; j < nnodes; j++) {
		rank[j] = (1 - DUMP) / nnodes + DUMP * sum[j]; 	
	}
	//}
}

void print(float *rank) {
	FILE *fout = fopen("ranks.txt", "w");
	for(unsigned i=0;i<nnodes;i++) {
		fprintf(fout, "%lf\n", rank[i]);
	}
	fclose(fout);
}

int main(int argc, char *argv[]) {
	start = omp_get_wtime();
	char *filename;
	if (argc > 4) {
		filename = argv[1];
		NUM_THREADS = strtoul(argv[2], NULL, 0);
		TILE_WIDTH = strtoul(argv[3], NULL, 0);
		CHUNK_SIZE = strtoul(argv[4], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec-relationships.txt";
		NUM_THREADS = 256;
		TILE_WIDTH = 1024;
		CHUNK_SIZE = 2048;
	}
	input(filename);
	return 0;
}
