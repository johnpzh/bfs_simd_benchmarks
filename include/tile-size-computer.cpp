#include <stdio.h>
#include <stdlib.h>


unsigned long get_num_tiles(
		unsigned nnodes,
		unsigned nedges,
		unsigned tile_size)
{
	unsigned long side_length;
	//unsigned side_length;
	if (nnodes % tile_size == 0) {
		side_length = nnodes / tile_size;
	} else {
		side_length = nnodes / tile_size + 1;
	}
	//printf("tile_size: %u, side_length: %u\n", tile_size, side_length);
	return side_length * side_length;
}

void compute(
		unsigned nnodes,
		unsigned nedges)
{
	unsigned tile_size = 64;
	unsigned long num_tiles;

	do {
		num_tiles = get_num_tiles(nnodes, nedges, tile_size);
		tile_size *= 2;
		//printf("tile_size: %u, num_tile: %lu\n", tile_size, num_tiles);//test
	} while (num_tiles > 4294967295L);
	tile_size /= 2;

	unsigned average_edge;
	do {
		average_edge = nedges / get_num_tiles(nnodes, nedges, tile_size);
		tile_size *= 2;
	} while (average_edge < 8);
	tile_size /= 2;
	unsigned min_tile_size = tile_size;
	
	unsigned side_length;
	do {
		if (nnodes % tile_size == 0) {
			side_length = nnodes / tile_size;
		} else {
			side_length = nnodes / tile_size + 1;
		}
		tile_size *= 2;
	} while (side_length > 64);
	tile_size /= 4;
	unsigned max_tile_size = tile_size;

	printf("min_tile_size: %u, max_tile_size: %u\n", min_tile_size, max_tile_size);
}

int main(int argc, char *argv[])
{
	if (argc < 3) {
		printf("Usage: ./tile-size-computer <num of vertices> <num of edges>\n");
		exit(1);
	}
	unsigned nnodes = strtoul(argv[1], NULL, 0);
	unsigned nedges = strtoul(argv[2], NULL, 0);
	compute(nnodes, nedges);

	return 0;
}
