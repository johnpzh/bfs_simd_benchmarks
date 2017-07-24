#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <immintrin.h>
using std::ifstream;
using std::string;
using std::getline;
using std::stringstream;
using std::cout;
using std::endl;

#define DUMP 0.85
#define MAX_NODES 1700000
#define MAX_EDGES 40000000

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

struct Graph {
	int n1[MAX_EDGES] __attribute__((aligned(ALIGNED_BYTES)));
	int n2[MAX_EDGES] __attribute__((aligned(ALIGNED_BYTES)));
	int nneibor[MAX_NODES] __attribute__((aligned(ALIGNED_BYTES)));
};

int nnodes, nedges;
Graph grah;
float rank[MAX_NODES] __attribute__((aligned(ALIGNED_BYTES)));
float sum[MAX_NODES] __attribute__((aligned(ALIGNED_BYTES)));
unsigned NUM_THREADS;

void page_rank();

void input(char filename[]) {
	//printf("data: %s\n", filename);
	FILE *fin = fopen(filename, "r");

	fscanf(fin, "%u %u", &nnodes, &nedges);
	for (unsigned i = 0; i < nnodes; ++i) {
		grah.nneibor[i] = 0;
	}
	for (unsigned i = 0; i < nedges; ++i) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		grah.n1[i] = n1;
		grah.n2[i] = n2;
		grah.nneibor[n1]++;
	}
	fclose(fin);

	// PageRank
	for (unsigned i = 0; i < 9; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		page_rank();
	}
}

void input2(string filename, int tilesize) {
	ifstream fin(filename.c_str());
	string line;
	getline(fin, line);
	stringstream sin(line);
	sin >> nnodes >> nedges;

	for(int i=0;i<nnodes;i++) {
		grah.nneibor[i] = 0;
	}

	int cur = 0;
	while(getline(fin, line)) {
		int n, n1, n2;
		stringstream sin1(line);
		while(sin1 >> n) {
			grah.n1[cur] = n / tilesize;
			grah.n2[cur] = n % tilesize;
			cur++;
		}
	}
	nedges = cur;
}

inline void get_seq_sum(unsigned index, unsigned frontier)
{
	for (unsigned i = index; i < frontier; ++i) {
		int n1 = grah.n1[i];
		int n2 = grah.n2[i];
#pragma omp atomic
		sum[n2] += rank[n1]/grah.nneibor[n1];
	}
}

void page_rank() {
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

#pragma omp parallel for num_threads(NUM_THREADS)
	//for(unsigned j=0;j<nedges;j++) {
//		int n1 = grah.n1[j];
//		int n2 = grah.n2[j];
//#pragma omp atomic
//		sum[n2] += rank[n1]/grah.nneibor[n1];
	for (unsigned i = 0; i < nedges; i += NUM_P_INT) {
		if (i + NUM_P_INT <= nedges) {
			// Full loaded SIMD lanes
			__m512i n1_v = _mm512_load_epi32(grah.n1 + i);
			__m512i n2_v = _mm512_load_epi32(grah.n2 + i);
			__m512i conflict_n2 = _mm512_conflict_epi32(n2_v);
			__mmask16 is_conflict = _mm512_cmpneq_epi32_mask(conflict_n2, zero_v);
			if (*((short *)(&is_conflict)) == 0) {
				// No conflicts
				__m512 rank_v = _mm512_i32gather_ps(n1_v, rank, sizeof(float));
				__m512i nneibor_vi = _mm512_i32gather_epi32(n1_v, grah.nneibor, sizeof(int));
				__m512 nneibor_v = _mm512_cvt_roundepi32_ps(nneibor_vi, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
				__m512 tmp_sum = _mm512_div_ps(rank_v, nneibor_v);
				__m512 sum_n2_v = _mm512_i32gather_ps(n2_v, sum, sizeof(float));
				tmp_sum = _mm512_add_ps(tmp_sum, sum_n2_v);
				_mm512_i32scatter_ps(sum, n2_v, tmp_sum, sizeof(float));
			} else {
				// Conflicts exists, then process sequentially
				get_seq_sum(i, i + NUM_P_INT);
			}
		} else {
			// Process remain sequentially
			get_seq_sum(i, nedges);
		}
	}
	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);

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
	if (argc > 2) {
		filename = argv[1];
		NUM_THREADS = strtoul(argv[2], NULL, 0);
	} else {
		filename = "/home/zpeng/benchmarks/data/soc-pokec-relationships.txt";
		NUM_THREADS = 256;
	}
	input(filename);
	double input_end = omp_get_wtime();
	//printf("input tims: %lf\n", input_end - input_start);
	//page_rank();
#ifdef ONEDEBUG
	print();
#endif
	return 0;
}
