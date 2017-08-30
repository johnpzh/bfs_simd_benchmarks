#!/bin/tcsh
#######PBS -l nodes=1:meltemi:ppn=256
#PBS -l nodes=1:hima:ppn=64
#PBS -l walltime=48:00:00
#PBS -N tiled_PageRank
#PBS -j oe
#PBS -m abe
cd /sciclone/pscr/zpeng01/benchmarks/test/tiled_naive_pageRank
./page_rank /sciclone/scr10/zpeng01/data/pokec/coo_tiled_bak/soc-pokec 1024
