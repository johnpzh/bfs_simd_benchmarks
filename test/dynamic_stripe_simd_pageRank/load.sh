#!/bin/tcsh
#PBS -l nodes=1:meltemi:ppn=256
#PBS -l walltime=2:00:00
#PBS -N stripe_PageRank
#PBS -j oe
#PBS -m abe
#PBS -o output.txt
#PBS -e error.txt
cd /sciclone/pscr/zpeng01/benchmarks/test/dynamic_stripe_naive_pageRank
./page_rank /sciclone/scr10/zpeng01/twitter/coo_tiled_bak/out.twitter 8192
