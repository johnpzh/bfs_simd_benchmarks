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
#include <algorithm>
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;
using std::vector;

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH; // Width of tile
unsigned SIDE_LENGTH; // Number of rows of tiles
unsigned NUM_TILES; // Number of tiles
unsigned ROW_STEP; // Number of rows of tiles in a Group

unsigned KCORE;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_ends, 
		unsigned *&tile_offsets,
		unsigned *&nneibor,
		int *&is_empty_tile) 
{
	//printf("data: %s\n", filename);
	//string prefix = string(filename) + "_untiled";
	string prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH) + "_reverse";
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	fclose(fin);
	if (NNODES % TILE_WIDTH) {
		SIDE_LENGTH = NNODES / TILE_WIDTH + 1;
	} else {
		SIDE_LENGTH = NNODES / TILE_WIDTH;
	}
	NUM_TILES = SIDE_LENGTH * SIDE_LENGTH;
	// Read tile Offsets
	fname = prefix + "-offsets";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	tile_offsets = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		//fscanf(fin, "%u", tile_offsets + i);
		unsigned offset;
		fscanf(fin, "%u", &offset);
		tile_offsets[i] = offset;
	}
	fclose(fin);
	// Read degrees
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	nneibor = (unsigned *) malloc(NNODES * sizeof(unsigned));
	for (unsigned i = 0; i < NNODES; ++i) {
		fscanf(fin, "%u", nneibor + i);
	}
	fclose(fin);

	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
	memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			if (tile_offsets[i] == tile_offsets[i + 1]) {
				is_empty_tile[i] = 1;
			}
		} else {
			if (tile_offsets[i] == NEDGES) {
				is_empty_tile[i] = 1;
			}
		}
	}
	NUM_THREADS = 64;
	unsigned edge_bound = NEDGES / NUM_THREADS;
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
		fscanf(fin, "%u %u\n", &NNODES, &NEDGES);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_ends[index] = n2;
	}

	fclose(fin);
}
}

void input_serial(char filename[], unsigned *&graph_heads, unsigned *&graph_ends)
{
	//printf("data: %s\n", filename);
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", filename);
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	for (unsigned i = 0; i < NEDGES; ++i) {
		unsigned head;
		unsigned end;
		fscanf(fin, "%u %u", &head, &end);
		--head;
		--end;
		graph_heads[i] = head;
		graph_ends[i] = end;
	}
	fclose(fin);
}

void convert_to_col_major(
						char *filename,
						unsigned *graph_heads, 
						unsigned *graph_ends, 
						unsigned *tile_offsets,
						unsigned *nneibor)
{
	unsigned *new_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	unsigned *new_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	unsigned *new_offsets = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	//unsigned step = 16;
	//unsigned step = 2;//test
	unsigned edge_index = 0;
	unsigned new_tile_id = 0;
	unsigned side_i = 0;

	printf("Converting...\n");

	for (side_i = 0; side_i + ROW_STEP <= SIDE_LENGTH; side_i += ROW_STEP) {
		for (unsigned col = 0; col < SIDE_LENGTH; ++col) {
			for (unsigned row = side_i; row < side_i + ROW_STEP; ++row) {
				unsigned tile_id = row * SIDE_LENGTH + col;
				unsigned bound_edge_i;
				if (NUM_TILES - 1 != tile_id) {
					bound_edge_i = tile_offsets[tile_id + 1];
				} else {
					bound_edge_i = NEDGES;
				}
				new_offsets[new_tile_id++] = edge_index;
				for (unsigned edge_i = tile_offsets[tile_id]; edge_i < bound_edge_i; ++edge_i) {
					new_heads[edge_index] = graph_heads[edge_i] + 1;
					new_ends[edge_index] = graph_ends[edge_i] + 1;
					++edge_index;
				}
			}
		}
	}
	if (side_i != SIDE_LENGTH) {
		for (unsigned col = 0; col < SIDE_LENGTH; ++col) {
			for (unsigned row = side_i; row < SIDE_LENGTH; ++row) {
				unsigned tile_id = row * SIDE_LENGTH + col;
				unsigned bound_edge_i;
				if (NUM_TILES - 1 != tile_id) {
					bound_edge_i = tile_offsets[tile_id + 1];
				} else {
					bound_edge_i = NEDGES;
				}
				new_offsets[new_tile_id++] = edge_index;
				for (unsigned edge_i = tile_offsets[tile_id]; edge_i < bound_edge_i; ++edge_i) {
					new_heads[edge_index] = graph_heads[edge_i] + 1;
					new_ends[edge_index] = graph_ends[edge_i] + 1;
					++edge_index;
				}
			}
		}
	}
	printf("Finally, edge_index: %u (NEDGES: %u), new_tile_id: %u (NUM_TILES: %u)\n", edge_index, NEDGES, new_tile_id, NUM_TILES);

	// Write to files
	string prefix = string(filename) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH) + "_reverse";
	NUM_THREADS = 64;
	unsigned edge_bound = NEDGES / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS)
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * edge_bound;
	string fname = prefix + "-" + to_string(tid);
	FILE *fout = fopen(fname.c_str(), "w");
	if (0 == tid) {
		fprintf(fout, "%u %u\n", NNODES, NEDGES);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		fprintf(fout, "%u %u\n", new_heads[index], new_ends[index]);
	}
	fclose(fout);
}
	printf("Main files done...\n");
	// Write offsets
	string fname = prefix + "-offsets";
	FILE *fout = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		fprintf(fout, "%u\n", new_offsets[i]);//Format: offset
	}
	fclose(fout);
	fname = prefix + "-nneibor";
	fout = fopen(fname.c_str(), "w");
	for (unsigned i = 0; i < NNODES; ++i) {
		fprintf(fout, "%u\n", nneibor[i]);
	}
	printf("Done.\n");

	//// test
	//fout = fopen("output.txt", "w");
	//fprintf(fout, "%u %u\n", NNODES, NEDGES);
	//for (unsigned i = 0; i < NEDGES; ++i) {
	//	fprintf(fout, "%u %u\n", new_heads[i], new_ends[i]);
	//}
	//fclose(fout);

	free(new_heads);
	free(new_ends);
}

int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	unsigned min_row_step;
	unsigned max_row_step;

	if (argc > 4) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
		//ROW_STEP = strtoul(argv[3], NULL, 0);
		min_row_step = strtoul(argv[3], NULL, 0);
		max_row_step = strtoul(argv[4], NULL, 0);
	} else {
		////filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		//filename = "/home/zpeng/benchmarks/data/skitter/out.skitter";
		//TILE_WIDTH = 1024;
		//ROW_STEP = 16;
		printf("Usage: ./kcore <data_file> <tile_width> <min_stripe_length> <max_stripe_length>\n");
		exit(1);
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_ends;
	unsigned *tile_offsets;
	unsigned *nneibor;
	int *is_empty_tile;

	input(
		filename, 
		graph_heads, 
		graph_ends, 
		tile_offsets,
		nneibor,
		is_empty_tile);

	for (ROW_STEP = min_row_step; ROW_STEP <= max_row_step; ROW_STEP *= 2) {
	convert_to_col_major(
						filename,
						graph_heads, 
						graph_ends, 
						tile_offsets,
						nneibor);
	}

	// Free memory
	free(graph_heads);
	free(graph_ends);
	free(tile_offsets);

	return 0;
}
