////////////////////////////////////////////////////////////
// Convert Adj format (used by Ligra) to SNAP format (edge list).
// Support only unweighted graph right now.
////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string>

using std::string;
using std::to_string;

void data_input(
		const string &in_filename,
		unsigned *&graph_vertices,
		unsigned *&graph_edges,
		unsigned &nnodes,
		unsigned &nedges)
{
	FILE *fin = fopen(in_filename.c_str(), "r");
	if (nullptr == fin) {
		printf("Error: cannot open file %s.\n", in_filename.c_str());
		exit(2);
	}
	// Read Number of Nodes, and Edges
	fscanf(fin, "%*s%u%u", &nnodes, &nedges);

	graph_vertices = (unsigned *) malloc(nnodes * sizeof(unsigned));
	graph_edges = (unsigned *) malloc(nedges * sizeof(unsigned));
	// Read vertices and edges
	for (unsigned i = 0; i < nnodes; ++i) {
		fscanf(fin, "%u", graph_vertices + i);
	}
	for (unsigned i = 0; i < nedges; ++i) {
		fscanf(fin, "%u", graph_edges + i);
	}

	fclose(fin);
}
void data_convert(
			unsigned *graph_vertices,
			unsigned *graph_edges,
			const unsigned &nnodes,
			const unsigned &nedges,
			unsigned *graph_heads,
			unsigned *graph_tails)
{
	for (unsigned vertex_i = 0; vertex_i < nnodes; ++vertex_i) {
		unsigned start_edge_i = graph_vertices[vertex_i];
		unsigned bound_edge_i;
		if (nnodes - 1 != vertex_i) {
			bound_edge_i = graph_vertices[vertex_i + 1];
		} else {
			bound_edge_i = nedges;
		}
		for (unsigned edge_i = start_edge_i; edge_i < bound_edge_i; ++edge_i) {
			unsigned tail_i = graph_edges[edge_i];
			graph_heads[edge_i] = vertex_i;
			graph_tails[edge_i] = tail_i;
		}
	}
}
void data_output(
			const string &out_filename,
			const unsigned &nnodes,
			const unsigned &nedges,
			unsigned *graph_heads,
			unsigned *graph_tails)
{
	FILE *fout = fopen(out_filename.c_str(), "w");

	fprintf(fout, "%u %u\n", nnodes, nedges);
	for (unsigned i = 0; i < nedges; ++i) {
		fprintf(fout, "%u %u\n", graph_heads[i] + 1, graph_tails[i] + 1);
	}
	fclose(fout);
}
int main(int argc, char *argv[])
{
	string in_filename;
	string out_filename;
	if (argc > 2) {
		//in_filename = to_string(argv[1]);
		//out_filename = to_string(argv[2]);
		in_filename = string(argv[1]);
		out_filename = string(argv[2]);
	} else {
		printf("Usage: ./adj2snap <input_file> <output_file>\n");
		exit(1);
	}
	unsigned nnodes;
	unsigned nedges;

	unsigned *graph_vertices = nullptr;
	unsigned *graph_edges = nullptr;

	data_input(
		in_filename,
		graph_vertices,
		graph_edges,
		nnodes,
		nedges);

	unsigned *graph_heads = (unsigned *) malloc(nedges * sizeof(unsigned));
	unsigned *graph_tails = (unsigned *) malloc(nedges * sizeof(unsigned));

	data_convert(
			graph_vertices,
			graph_edges,
			nnodes,
			nedges,
			graph_heads,
			graph_tails);

	data_output(
			out_filename,
			nnodes,
			nedges,
			graph_heads,
			graph_tails);

	free(graph_vertices);
	free(graph_edges);
	free(graph_heads);
	free(graph_tails);
}
