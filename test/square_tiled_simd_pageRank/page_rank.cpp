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

#define DUMP 0.85
#define MAX_NODES 1700000
#define MAX_EDGES 40000000

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

struct Graph {
	//int n1[MAX_EDGES];
	//int n2[MAX_EDGES];
	int nneibor[MAX_NODES];
};

int nnodes, nedges;
Graph grah;
float rank[MAX_NODES];
float sum[MAX_NODES];
unsigned NUM_THREADS;
unsigned tile_width;
unsigned CHUNK_SIZE;

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *tops, unsigned *offsets, unsigned num_tiles);

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
//	memset(grah.nneibor, 0, sizeof(grah.nneibor));
//	unsigned num_tiles;
//	//unsigned long long num_tiles;
//	unsigned side_length;
//	if (nnodes % tile_width) {
//		side_length = nnodes / tile_width + 1;
//	} else {
//		side_length = nnodes / tile_width;
//	}
//	num_tiles = side_length * side_length;
//	if (nedges < num_tiles) {
//		fprintf(stderr, "Error: tile size is too small.\n");
//		exit(2);
//	}
//	//unsigned max_top = nedges / num_tiles * 16;
//	unsigned max_top = tile_width * tile_width / 128;
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
//		unsigned n1_id = n1 / tile_width;
//		unsigned n2_id = n2 / tile_width;
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
//		grah.nneibor[n1]++;
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
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", filename);
		exit(1);
	}

	fscanf(fin, "%u %u", &nnodes, &nedges);
	memset(grah.nneibor, 0, sizeof(grah.nneibor));
	unsigned num_tiles;
	//unsigned long long num_tiles;
	unsigned side_length;
	if (nnodes % tile_width) {
		side_length = nnodes / tile_width + 1;
	} else {
		side_length = nnodes / tile_width;
	}
	num_tiles = side_length * side_length;
	vector< vector<unsigned> > tiles_n1v;
	tiles_n1v.resize(num_tiles);
	vector< vector<unsigned> > tiles_n2v;
	tiles_n2v.resize(num_tiles);
	//unsigned tile_size = tile_width * tile_width;
	//if (nedges < num_tiles) {
	//	fprintf(stderr, "Error: tile size is too small.\n");
	//	exit(2);
	//}
	unsigned *tops = (unsigned *) _mm_malloc(num_tiles * sizeof(unsigned), ALIGNED_BYTES);
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

		//unsigned *top = tops + tile_id;
		//if (*top == tile_size) {
		//	fprintf(stderr, "Error: the tile %u is full.\n", tile_id);
		//	exit(1);
		//}
		//tiles_n1[*top + offsets[tile_id]] = n1;
		//tiles_n2[*top + offsets[tile_id]] = n2;
		//(*top)++;
		grah.nneibor[n1]++;
		tiles_n1v[tile_id].push_back(n1);
		tiles_n2v[tile_id].push_back(n2);
	}
	unsigned *tiles_n1 = (unsigned *) _mm_malloc(nedges * sizeof(unsigned), ALIGNED_BYTES);
	unsigned *tiles_n2 = (unsigned *) _mm_malloc(nedges * sizeof(unsigned), ALIGNED_BYTES);
	unsigned *offsets = (unsigned *) _mm_malloc(num_tiles * sizeof(unsigned), ALIGNED_BYTES);
	unsigned offset = 0;
	unsigned index = 0;
	for (unsigned i = 0; i < num_tiles; ++i) {
		unsigned size = tiles_n1v[i].size();
		offsets[i] = offset;
		offset += size;
		for (unsigned j = 0; j < size; ++j) {
			unsigned n1 = tiles_n1v[i][j];
			unsigned n2 = tiles_n2v[i][j];
			tiles_n1[index] = n1;
			tiles_n2[index] = n2;
			++index;
		}
	}
	// Clean the vectors for saving memory
	for (unsigned i = 0; i < num_tiles; ++i) {
		tiles_n1v[i].clear();
		tiles_n2v[i].clear();
	}
	tiles_n1v.clear();
	tiles_n2v.clear();
	fclose(fin);

	// PageRank
	for (unsigned i = 0; i < 9; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		page_rank(tiles_n1, tiles_n2, tops, offsets, num_tiles);
	}

	// Free memory
	_mm_free(tiles_n1);
	_mm_free(tiles_n2);
	_mm_free(offsets);
	_mm_free(tops);
}

inline void get_seq_sum(unsigned *n1s, unsigned *n2s, unsigned index, unsigned frontier)
{
	for (unsigned i = index; i < frontier; ++i) {
		unsigned n1 = n1s[i];
		unsigned n2 = n2s[i];
//#pragma omp atomic
		sum[n2] += rank[n1]/grah.nneibor[n1];
	}
}

void page_rank(unsigned *tiles_n1, unsigned *tiles_n2, unsigned *tops, unsigned *offsets, unsigned num_tiles) {
	const __m512i one_v = _mm512_set1_epi32(1);
	const __m512i zero_v = _mm512_set1_epi32(0);
	const __m512i minusone_v = _mm512_set1_epi32(-1);
//#pragma omp parallel for num_threads(256)
	for(unsigned i=0;i<nnodes;i++) {
		rank[i] = 1.0;
		sum[i] = 0.0;
	}

	//for(int i=0;i<10;i++) {
	double start_time = omp_get_wtime();

#pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic, CHUNK_SIZE)
//#pragma omp parallel for num_threads(NUM_THREADS)
	for (unsigned i = 0; i < num_tiles; ++i) {
		unsigned top = tops[i];
		unsigned j = 0;
		//unsigned frontier = top - NUM_P_INT;
		unsigned frontier;
		if (top <= NUM_P_INT) {
			frontier = 0;
		} else {
			frontier = top - NUM_P_INT;
		}
		for (; j < frontier; j += NUM_P_INT) {
			//if (j + NUM_P_INT <= top) {
				// Full loaded SIMD lanes
			__m512i n1_v = _mm512_load_epi32(tiles_n1 + offsets[i] + j);
			__m512i n2_v = _mm512_load_epi32(tiles_n2 + offsets[i] + j);
			__m512i conflict_n2 = _mm512_conflict_epi32(n2_v);
			__mmask16 is_conflict = _mm512_cmpneq_epi32_mask(conflict_n2, zero_v);
			if (*((short *)(&is_conflict)) == 0) {
				// No conflicts
				__m512 rank_v = _mm512_i32gather_ps(n1_v, rank, sizeof(float));
				__m512i nneibor_vi = _mm512_i32gather_epi32(n1_v, grah.nneibor, sizeof(int));
				__m512 nneibor_v = _mm512_cvtepi32_ps(nneibor_vi);
				__m512 tmp_sum = _mm512_div_ps(rank_v, nneibor_v);
				__m512 sum_n2_v = _mm512_i32gather_ps(n2_v, sum, sizeof(float));
				tmp_sum = _mm512_add_ps(tmp_sum, sum_n2_v);
				_mm512_i32scatter_ps(sum, n2_v, tmp_sum, sizeof(float));
			} else {
				// Conflicts exists, then process sequentially
				get_seq_sum(tiles_n1 + offsets[i], tiles_n2 + offsets[i], j, j + NUM_P_INT);
			}
		}
		// Process remain sequentially
		get_seq_sum(tiles_n1 + offsets[i], tiles_n2 + offsets[i], j, top);
	}

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
	//printf("%u %lf\n", tile_width, end_time - start_time);
	//printf("%u %lf\n", CHUNK_SIZE, end_time - start_time);

#pragma omp parallel for num_threads(256)
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
	if (argc > 4) {
		filename = argv[1];
		NUM_THREADS = strtoul(argv[2], NULL, 0);
		tile_width = strtoul(argv[3], NULL, 0);
		CHUNK_SIZE = strtoul(argv[4], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/soc-pokec-relationships.txt";
		NUM_THREADS = 256;
		tile_width = 1024;
		CHUNK_SIZE = 2048;
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
