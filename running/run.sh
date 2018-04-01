#! /usr/bin/bash
if [[ $# -lt 6 ]]; then
	echo "Usage: ./run.sh <mark> <data_file> <bfs_tile_size> <bfs_stripe_length> <pr_tile_size> <pr_stripe_length>"
	exit
fi

apps_dir=/home/zpeng/benchmarks/test/
mark=$1
data_file=$2
bfs_tile_size=$3
bfs_stripe_length=$4
pr_tile_size=$5
pr_stripe_length=$6

# BFS
${apps_dir}/bfs_simd/bfs $data_file $bfs_tile_size $bfs_stripe_length > output_$(date +%Y%m%d-%H%M%S)_bfs_${mark}.txt

# PageRank
${apps_dir}/pageRank_simd/page_rank $data_file $pr_tile_size $pr_stripe_length > output_$(date +%Y%m%d-%H%M%S)_pageRank_${mark}.txt

# CC
${apps_dir}/betweennessCentrality_simd/bc $data_file $bfs_tile_size $bfs_stripe_length > output_$(date +%Y%m%d-%H%M%S)_bfs_${mark}.txt

# MIS

# SSSP
${apps_dir}/sssp_simd/sssp ${data_file}_weighted $bfs_tile_size $bfs_stripe_length > output_$(date +%Y%m%d-%H%M%S)_sssp_${mark}.txt

# BC
