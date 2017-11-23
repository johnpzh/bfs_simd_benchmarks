#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer_opadd.h>
#include <omp.h>
#include <string>
#include <unistd.h>

using std::string;
using std::to_string;

struct Vertex {
	unsigned *out_neighbors;
	unsigned out_degree;

	unsigned get_out_neighbor(unsigned i) {
		return out_neighbors[i];
	}
	unsigned get_out_degree() {
		return out_degree;
	}
};


unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH;
unsigned SIDE_LENGTH;
unsigned NUM_TILES;
unsigned ROW_STEP;

void set_nworkers(int n)
{
	printf("@set_nworkers: ");
	__cilkrts_end_cilk();
	switch (__cilkrts_set_param("nworkers", to_string(NUM_THREADS).c_str())) {
		case __CILKRTS_SET_PARAM_SUCCESS:
			printf("set worker successfully to %d (%d).\n", n, __cilkrts_get_nworkers());
			return;
		case __CILKRTS_SET_PARAM_UNIMP:
			printf("Unimplemented parameter.\n");
			break;
		case __CILKRTS_SET_PARAM_XRANGE:
			printf("Parameter value out of range.\n");
			break;
		case __CILKRTS_SET_PARAM_INVALID:
			printf("Invalid parameter value.\n");
			break;
		case __CILKRTS_SET_PARAM_LATE:
			printf("Too late to change parameter value.\n");
			break;
	}
	exit(1);
}

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

double offset_time1 = 0;
double offset_time2 = 0;
double degree_time = 0;
double frontier_tmp_time = 0;
double refine_time = 0;
double arrange_time = 0;
double run_time = 0;
unsigned *BFS_kernel(
				//unsigned *graph_vertices,
				Vertex *graph_vertices_info,
				unsigned *graph_edges,
				//unsigned *h_graph_degrees,
				unsigned *parents,
				unsigned *&frontier,
				unsigned &frontier_size)
{
	// From frontier, get the degrees (para_for)
	double time_now = omp_get_wtime(); 
	unsigned *degrees = (unsigned *) malloc(sizeof(unsigned) *  frontier_size);
	Vertex *frontier_vertices = (Vertex *) malloc(sizeof(Vertex) * frontier_size);
	unsigned new_frontier_size = 0;
	cilk::reducer< cilk::op_add<unsigned> > parallel_size(0);
//#pragma omp parallel for schedule(dynamic) reduction(+: new_frontier_size)
//#pragma omp parallel for reduction(+: new_frontier_size)
	cilk_for (unsigned i = 0; i < frontier_size; ++i) {
		unsigned start = frontier[i];
		Vertex v = graph_vertices_info[start];
		degrees[i] = v.get_out_degree();
		//new_frontier_size += degrees[i];
		*parallel_size += degrees[i];
		frontier_vertices[i] = v;
	}
	new_frontier_size = parallel_size.get_value();
	if (0 == new_frontier_size) {
		free(degrees);
		frontier_size = 0;
		return NULL;
	}
	degree_time += omp_get_wtime() - time_now;

	// From degrees, get the offset (stored in degrees) (block_para_for)
	// TODO: blocked parallel for
	//unsigned *offsets = (unsigned *) malloc(sizeof(unsigned) * frontier_size);
	time_now = omp_get_wtime();
	unsigned offset_sum = 0;
	for (unsigned i = 0; i < frontier_size; ++i) {
		unsigned tmp = degrees[i];
		degrees[i] = offset_sum;
		offset_sum += tmp;
	}
	offset_time1 += omp_get_wtime() - time_now;
	//offsets[0] = 0;
	//for (unsigned i = 1; i < frontier_size; ++i) {
	//	offsets[i] = offsets[i - 1] + degrees[i - 1];
	//}

	// From offset, get active vertices (para_for)
	time_now = omp_get_wtime();
	unsigned *new_frontier_tmp = (unsigned *) malloc(sizeof(unsigned) * new_frontier_size);
//#pragma omp parallel for schedule(dynamic)
//#pragma omp parallel for
//	for (unsigned i = 0; i < frontier_size; ++i) {}
//#pragma omp parallel num_threads(NUM_THREADS)
//{
//#pragma omp single
	cilk_for (unsigned i = 0; i < frontier_size; ++i) {
		//printf("wid: %d\n", __cilkrts_get_worker_number());//test
		Vertex start = frontier_vertices[i];
		unsigned start_id = frontier[i];
		unsigned offset = degrees[i];
		//unsigned size = 0;
		// no speedup
		unsigned out_degree = start.out_degree;
		if (out_degree > 1000) {
			cilk_for (unsigned i = 0; i < out_degree; ++i) {
				unsigned end = start.get_out_neighbor(i);
				if ((unsigned)-1 == parents[end]) {
					bool unvisited = __sync_bool_compare_and_swap(parents + end, (unsigned) -1, start_id); //update parents
					if (unvisited) {
						//new_frontier_tmp[offset + size++] = end;
						new_frontier_tmp[offset + i] = end;
					} else {
						new_frontier_tmp[offset + i] = (unsigned) -1;
						//new_frontier_tmp[offset + size++] = (unsigned) -1;
					}
				} else {
					new_frontier_tmp[offset + i] = (unsigned) -1;
				}
			}
		} else {
			for (unsigned i = 0; i < out_degree; ++i) {
				unsigned end = start.get_out_neighbor(i);
				if ((unsigned)-1 == parents[end]) {
					bool unvisited = __sync_bool_compare_and_swap(parents + end, (unsigned) -1, start_id); //update parents
					if (unvisited) {
						//new_frontier_tmp[offset + size++] = end;
						new_frontier_tmp[offset + i] = end;
					} else {
						new_frontier_tmp[offset + i] = (unsigned) -1;
						//new_frontier_tmp[offset + size++] = (unsigned) -1;
					}
				} else {
					new_frontier_tmp[offset + i] = (unsigned) -1;
				}
			}
		}
		// end no speedup
		//unsigned *bound_edge_i = start.out_neighbors + start.out_degree;
		//if (start.out_degree <= 1000) {
		//for (unsigned *edge_i = start.out_neighbors; edge_i != bound_edge_i; ++edge_i) {
		//	unsigned end = *edge_i;
		//	bool unvisited = __sync_bool_compare_and_swap(parents + end, (unsigned) -1, start_id); //update parents
		//	if (unvisited) {
		//		new_frontier_tmp[offset + size++] = end;
		//	} else {
		//		new_frontier_tmp[offset + size++] = (unsigned) -1;
		//	}
		//}
		//} else {
		//cilk_for (unsigned *edge_i = start.out_neighbors; edge_i != bound_edge_i; ++edge_i) {
		//	unsigned end = *edge_i;
		//	bool unvisited = __sync_bool_compare_and_swap(parents + end, (unsigned) -1, start_id); //update parents
		//	if (unvisited) {
		//		new_frontier_tmp[offset + size++] = end;
		//	} else {
		//		new_frontier_tmp[offset + size++] = (unsigned) -1;
		//	}
		//}
		//}
	}
//}
//	for (unsigned i = 0; i < frontier_size; ++i) {
//		unsigned start = frontier[i];
//		//unsigned offset = offsets[i];
//		unsigned offset = degrees[i];
//		unsigned size = 0;
//		unsigned bound_edge_i = graph_vertices[start] + h_graph_degrees[start];
//		for (unsigned edge_i = graph_vertices[start]; edge_i < bound_edge_i; ++edge_i) {
//			unsigned end = graph_edges[edge_i];
//			bool unvisited = __sync_bool_compare_and_swap(parents + end, (unsigned) -1, start); //update parents
//			if (unvisited) {
//				new_frontier_tmp[offset + size++] = end;
//			} else {
//				new_frontier_tmp[offset + size++] = (unsigned) -1;
//			}
//		}
//	}
	frontier_tmp_time += omp_get_wtime() - time_now;


	// Refine active vertices, removing visited and redundant (block_para_for)
	//unsigned block_size = new_frontier_size / NUM_THREADS;
	time_now = omp_get_wtime();
	unsigned block_size = 1024 * 2;
	//unsigned num_blocks = new_frontier_size % block_size == 0 ? new_frontier_size/block_size : new_frontier_size/block_size + 1;
	unsigned num_blocks = (new_frontier_size - 1)/block_size + 1;

	unsigned *nums_in_blocks = NULL;
	if (num_blocks > 1) {
		//unsigned *nums_in_blocks = (unsigned *) malloc(sizeof(unsigned) * NUM_THREADS);
		nums_in_blocks = (unsigned *) malloc(sizeof(unsigned) * num_blocks);
		//unsigned new_frontier_size_tmp = 0;
		parallel_size.set_value(0);
//#pragma omp parallel for schedule(dynamic) reduction(+: new_frontier_size_tmp)
//#pragma omp parallel for reduction(+: new_frontier_size_tmp)
		cilk_for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
			unsigned offset = block_i * block_size;
			unsigned bound;
			if (num_blocks - 1 != block_i) {
				bound = offset + block_size;
			} else {
				bound = new_frontier_size;
			}
			//unsigned size = 0;
			unsigned base = offset;
			for (unsigned end_i = offset; end_i < bound; ++end_i) {
				if ((unsigned) - 1 != new_frontier_tmp[end_i]) {
					new_frontier_tmp[base++] = new_frontier_tmp[end_i];
				}
				//unsigned end = new_frontier_tmp[end_i];
				//if (parents[end] == (unsigned) -1) {
				//	new_frontier_tmp[offset + size]  = end;
				//	++size;
				//}
			}
			nums_in_blocks[block_i] = base - offset;
			//new_frontier_size_tmp += nums_in_blocks[block_i];
			*parallel_size += nums_in_blocks[block_i];
		}
		//new_frontier_size = new_frontier_size_tmp;
		new_frontier_size = parallel_size.get_value();
	} else {
		unsigned base = 0;
		for (unsigned i = 0; i < new_frontier_size; ++i) {
			if ((unsigned) -1 != new_frontier_tmp[i]) {
				new_frontier_tmp[base++] = new_frontier_tmp[i];
			}
		}
		new_frontier_size = base;
	}
	refine_time += omp_get_wtime() - time_now;
	
	if (0 == new_frontier_size) {
		//free(offsets);
		free(frontier_vertices);
		free(degrees);
		free(new_frontier_tmp);
		if (nums_in_blocks) {
			free(nums_in_blocks);
		}
		frontier_size = new_frontier_size;
		return NULL;
	}

	// Get the final new frontier
	//unsigned *offsets_b = (unsigned *) malloc(sizeof(unsigned) * num_blocks);
	//offsets_b[0] = 0;
	time_now = omp_get_wtime();
	unsigned *new_frontier = (unsigned *) malloc(sizeof(unsigned) * new_frontier_size);
	if (num_blocks > 1) {
		//TODO: blocked parallel for
		double time_now = omp_get_wtime();
		offset_sum = 0;
		for (unsigned i = 0; i < num_blocks; ++i) {
			unsigned tmp = nums_in_blocks[i];
			nums_in_blocks[i] = offset_sum;
			offset_sum += tmp;
			//offsets_b[i] = offsets_b[i - 1] + nums_in_blocks[i - 1];
		}
		offset_time2 += omp_get_wtime() - time_now;
		//#pragma omp parallel for schedule(dynamic)
//#pragma omp parallel for
		cilk_for (unsigned block_i = 0; block_i < num_blocks; ++block_i) {
			//unsigned offset = offsets_b[block_i];
			unsigned offset = nums_in_blocks[block_i];
			unsigned bound;
			if (num_blocks - 1 != block_i) {
				bound = nums_in_blocks[block_i + 1];
			} else {
				bound = new_frontier_size;
			}
			//unsigned bound = offset + nums_in_blocks[block_i];
			unsigned base = block_i * block_size;
			for (unsigned i = offset; i < bound; ++i) {
				new_frontier[i] = new_frontier_tmp[base++];
			}
		}
	} else {
		unsigned base = 0;
		for (unsigned i = 0; i < new_frontier_size; ++i) {
			new_frontier[i] = new_frontier_tmp[base++];
		}
	}
	arrange_time += omp_get_wtime() - time_now;

	// Return the results
	//free(offsets);
	//free(offsets_b);
	free(frontier_vertices);
	free(degrees);
	free(new_frontier_tmp);
	if (nums_in_blocks) {
		free(nums_in_blocks);
	}
	//free(frontier);
	//frontier = new_frontier;
	frontier_size = new_frontier_size;
	return new_frontier;
}
void BFS(
		//unsigned *graph_vertices,
		Vertex *graph_vertices_info,
		unsigned *graph_edges,
		unsigned *h_graph_degrees,
		const unsigned &source,
		int *h_cost)
{

	//omp_set_num_threads(NUM_THREADS);
	//set_nworkers(NUM_THREADS);
	//__cilkrts_end_cilk();
	////__cilkrts_set_param("nworkers", to_string(NUM_THREADS).c_str());
	//switch (__cilkrts_set_param("nworkers", "128")) {
	//	case __CILKRTS_SET_PARAM_SUCCESS:
	//		printf("set worker successfully.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_UNIMP:
	//		printf("Unimplemented parameter.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_XRANGE:
	//		printf("Parameter value out of range.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_INVALID:
	//		printf("Invalid parameter value.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_LATE:
	//		printf("Too late to change parameter value.\n");
	//		break;
	//}
	//printf("nworkers: %d\n", __cilkrts_get_nworkers());
	unsigned frontier_size = 1;
	unsigned *frontier = (unsigned *) malloc(sizeof(unsigned) * frontier_size);
	frontier[0] = source;
	unsigned *parents = (unsigned *) malloc(sizeof(unsigned) * NNODES);
//#pragma omp parallel for num_threads(256)
	cilk_for (unsigned i = 0; i < NNODES; ++i) {
		parents[i] = (unsigned) -1; // means unvisited yet
	}
	parents[source] = source;
	double start_time = omp_get_wtime();
	while (frontier_size != 0) {
		// BFS_Kernel get new frontier and size
		unsigned *new_frontier = BFS_kernel(
				//graph_vertices,
				graph_vertices_info,
				graph_edges,
				//h_graph_degrees,
				parents,
				frontier,
				frontier_size);
		free(frontier);
		frontier = new_frontier;

		// Update distance and visited flag for new frontier
//#pragma omp parallel for
		cilk_for (unsigned i = 0; i < frontier_size; ++i) {
			unsigned end = frontier[i];
			unsigned start = parents[end];
			h_cost[end] = h_cost[start] + 1;
		}


	}
	double end_time = omp_get_wtime();
	printf("%d %lf\n", NUM_THREADS, run_time = (end_time - start_time));
	//free(frontier);
	free(parents);

}
///////////////////////////////////////////////////////////////////////////////
// Apply BFS on a Graph
///////////////////////////////////////////////////////////////////////////////
void input( int argc, char** argv) 
{
	//__cilkrts_end_cilk();
	//switch (__cilkrts_set_param("nworkers", "256")) {
	//	case __CILKRTS_SET_PARAM_SUCCESS:
	//		printf("set worker successfully.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_UNIMP:
	//		printf("Unimplemented parameter.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_XRANGE:
	//		printf("Parameter value out of range.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_INVALID:
	//		printf("Invalid parameter value.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_LATE:
	//		printf("Too late to change parameter value.\n");
	//		break;
	//}
	//printf("@input:357 : nworkers: %d\n", __cilkrts_get_nworkers());
	char *input_f;
	ROW_STEP = 16;
	//ROW_STEP = 2;
	
	if(argc < 2){
		input_f = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		//TILE_WIDTH = 1024;
	} else {
		input_f = argv[1];
		//TILE_WIDTH = strtoul(argv[2], NULL, 0);
	}

	/////////////////////////////////////////////////////////////////////
	// Input real dataset
	/////////////////////////////////////////////////////////////////////
	string prefix = string(input_f) + "_untiled";
	//string prefix = string(input_f) + "_coo-tiled-" + to_string(TILE_WIDTH);
	//string prefix = string(input_f) + "_col-16-coo-tiled-" + to_string(TILE_WIDTH);
	//string prefix = string(input_f) + "_col-2-coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	fclose(fin);
	//if (NNODES % TILE_WIDTH) {
	//	SIDE_LENGTH = NNODES / TILE_WIDTH + 1;
	//} else {
	//	SIDE_LENGTH = NNODES / TILE_WIDTH;
	//}
	//NUM_TILES = SIDE_LENGTH * SIDE_LENGTH;
	//// Read tile Offsets
	//fname = prefix + "-offsets";
	//fin = fopen(fname.c_str(), "r");
	//if (!fin) {
	//	fprintf(stderr, "cannot open file: %s\n", fname.c_str());
	//	exit(1);
	//}
	//unsigned *tile_offsets = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	//for (unsigned i = 0; i < NUM_TILES; ++i) {
	//	fscanf(fin, "%u", tile_offsets + i);
	//}
	//fclose(fin);
	unsigned *h_graph_starts = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
	unsigned *h_graph_ends = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
	unsigned *h_graph_degrees = (unsigned *) malloc(sizeof(unsigned) * NNODES);
	memset(h_graph_degrees, 0, sizeof(unsigned) * NNODES);
	//int *is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
	//memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);

	// Read degrees
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	for (unsigned i = 0; i < NNODES; ++i) {
		fscanf(fin, "%u", h_graph_degrees + i);
	}
	fclose(fin);

	NUM_THREADS = 64;
	unsigned edge_bound = NEDGES / NUM_THREADS;
	//__cilkrts_end_cilk();
	//switch (__cilkrts_set_param("nworkers", "256")) {
	//	case __CILKRTS_SET_PARAM_SUCCESS:
	//		printf("set worker successfully.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_UNIMP:
	//		printf("Unimplemented parameter.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_XRANGE:
	//		printf("Parameter value out of range.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_INVALID:
	//		printf("Invalid parameter value.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_LATE:
	//		printf("Too late to change parameter value.\n");
	//		break;
	//}
	//printf("@input:418 : nworkers: %d\n", __cilkrts_get_nworkers());
//#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
//{
		//unsigned tid = omp_get_thread_num();
	cilk_for (int tid = 0; tid < NUM_THREADS; ++tid) {
		unsigned offset = tid * edge_bound;
		string fname = prefix + "-" + to_string(tid);
		FILE *fin = fopen(fname.c_str(), "r");
		if (!fin) {
			fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
			exit(1);
		}
		if (0 == tid) {
			fscanf(fin, "%u %u", &NNODES, &NEDGES);
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
			h_graph_starts[index] = n1;
			h_graph_ends[index] = n2;
		}
		fclose(fin);
	}

//}
	//__cilkrts_end_cilk();
	//switch (__cilkrts_set_param("nworkers", "256")) {
	//	case __CILKRTS_SET_PARAM_SUCCESS:
	//		printf("set worker successfully.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_UNIMP:
	//		printf("Unimplemented parameter.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_XRANGE:
	//		printf("Parameter value out of range.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_INVALID:
	//		printf("Invalid parameter value.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_LATE:
	//		printf("Too late to change parameter value.\n");
	//		break;
	//}
	//printf("@input:448 : nworkers: %d\n", __cilkrts_get_nworkers());
	// CSR
	Vertex *graph_vertices_info = (Vertex *) malloc(sizeof(Vertex) * NNODES);
	//unsigned *graph_vertices = (unsigned *) malloc(sizeof(unsigned) * NNODES);
	unsigned *graph_edges = (unsigned *) malloc(sizeof(unsigned) * NEDGES);
	unsigned edge_start = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		//graph_vertices[i] = edge_start;
		graph_vertices_info[i].out_neighbors = graph_edges + edge_start;
		graph_vertices_info[i].out_degree = h_graph_degrees[i];
		edge_start += h_graph_degrees[i];
	}
	memcpy(graph_edges, h_graph_ends, sizeof(unsigned) * NEDGES);
	free(h_graph_starts);
	free(h_graph_ends);


	// End Input real dataset
	/////////////////////////////////////////////////////////////////////

	//int *h_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	//int *h_updating_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	//int *h_graph_visited = (int*) malloc(sizeof(int)*NNODES);
	int *h_cost = (int*) malloc(sizeof(int)*NNODES);
	//int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	//int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned source = 0;

	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	printf("Input finished: %s\n", input_f);
	unsigned run_count = 9;
#else
	unsigned run_count = 9;
#endif
	// BFS
	//for (unsigned ROW_STEP = 1; ROW_STEP < 10000; ROW_STEP *= 2) {
	//printf("ROW_STEP: %u\n", ROW_STEP);
	//unsigned ROW_STEP = 16;
	//ROW_STEP = 16;
	//ROW_STEP = 2;//test
	for (unsigned i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
#ifndef ONEDEBUG
		//sleep(10);
#endif
		// Re-initializing
		//memset(h_graph_mask, 0, sizeof(int)*NNODES);
		//h_graph_mask[source] = 1;
		//memset(h_updating_graph_mask, 0, sizeof(int)*NNODES);
		//memset(h_graph_visited, 0, sizeof(int)*NNODES);
		//h_graph_visited[source] = 1;
		for (unsigned i = 0; i < NNODES; ++i) {
			h_cost[i] = -1;
		}
		h_cost[source] = 0;
		//memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
		//is_active_side[0] = 1;
		//memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);

		//BFS(\
		//	h_graph_starts,\
		//	h_graph_ends,\
		//	h_graph_mask,\
		//	h_updating_graph_mask,\
		//	h_graph_visited,\
		//	h_cost,\
		//	tile_offsets,
		//	is_empty_tile,\
		//	is_active_side,\
		//	is_updating_active_side);
		offset_time1 = offset_time2 = 0;
		degree_time = 0;
		frontier_tmp_time = 0;
		refine_time = 0;
		arrange_time = 0;
		run_time = 0;
		printf("nworkers: %d\n", __cilkrts_get_nworkers());
		BFS(
			//graph_vertices,
			graph_vertices_info,
			graph_edges,
			h_graph_degrees,
			source,
			h_cost);
		auto percent = [] (double t) {return t/run_time*100;};
		printf("offset_time1: %f\noffset_time2: %f, sum: %f (%.1f%%)\n", offset_time1, offset_time2, offset_time1+offset_time2, percent(offset_time1+offset_time2));
		printf("degree_time: %f (%.1f%%)\n", degree_time, percent(degree_time));
		printf("frontier_tmp_time: %f (%.1f%%)\nrefine_time: %f (%.1f%%)\narrange_time: %f (%.1f%%)\n", frontier_tmp_time, percent(frontier_tmp_time), refine_time, percent(refine_time), arrange_time, percent(arrange_time));

		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
#ifdef ONEDEBUG
		printf("Thread %u finished.\n", NUM_THREADS);
#endif
	}
	//}
	fclose(time_out);

	//Store the result into a file

#ifdef ONEDEBUG
	///////////////////////////////////////////////
	// Store in a single file
	//FILE *foutput = fopen("path/path.txt", "w");
	//for (unsigned i = 0; i < NNODES; ++i) {
	//	fprintf(foutput, "%d) cost:%d\n", i, h_cost[i]);
	//}
	//fclose(foutput);
	//exit(0);
	//////////////////////////////////////////////
	NUM_THREADS = 64;
	//omp_set_num_threads(NUM_THREADS);
	unsigned num_lines = NNODES / NUM_THREADS;
//#pragma omp parallel
//{
	//unsigned tid = omp_get_thread_num();
	cilk_for (int tid = 0; tid < NUM_THREADS; ++tid) {
		unsigned offset = tid * num_lines;
		string file_prefix = "path/path";
		string file_name = file_prefix + to_string(tid) + ".txt";
		FILE *fpo = fopen(file_name.c_str(), "w");
		if (!fpo) {
			fprintf(stderr, "Error: cannot open file %s.\n", file_name.c_str());
			exit(1);
		}
		unsigned bound_index;
		if (tid != NUM_THREADS - 1) {
			bound_index = offset + num_lines;
		} else {
			bound_index = NNODES;
		}
		for (unsigned index = offset; index < bound_index; ++index) {
			fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
		}

		fclose(fpo);
	}
//}
#endif

	// cleanup memory
	//free( h_graph_starts);
	//free( h_graph_ends);
	//free( graph_vertices);
	free( graph_vertices_info);
	free( graph_edges);
	//free( h_graph_mask);
	//free( h_updating_graph_mask);
	//free( h_graph_visited);
	free( h_cost);
	//free( tile_offsets);
	//free( is_empty_tile);
	//free( is_active_side);
	//free( is_updating_active_side);
}
///////////////////////////////////////////////////////////////////////////////
// Main Program
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	//switch (__cilkrts_set_param("nworkers", "256")) {
	//	case __CILKRTS_SET_PARAM_SUCCESS:
	//		printf("set worker successfully.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_UNIMP:
	//		printf("Unimplemented parameter.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_XRANGE:
	//		printf("Parameter value out of range.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_INVALID:
	//		printf("Invalid parameter value.\n");
	//		break;
	//	case __CILKRTS_SET_PARAM_LATE:
	//		printf("Too late to change parameter value.\n");
	//		break;
	//}
	//printf("@main: nworkers: %d\n", __cilkrts_get_nworkers());
	start = omp_get_wtime();
	input( argc, argv);
}

