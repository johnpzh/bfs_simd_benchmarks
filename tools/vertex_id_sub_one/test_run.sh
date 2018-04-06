#! /usr/bin/bash

# road_usa_undirected
make clean
make undirected=1
./page_rank /sciclone/scr-mlt/zpeng01/road_usa/road_usa

# road_usa_weighted_undirected
make clean
make weighted=1 undirected=1
./page_rank /sciclone/scr-mlt/zpeng01/road_usa/road_usa_weighted
