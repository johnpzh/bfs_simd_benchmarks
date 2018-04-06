#! /usr/bin/bash
if [[ $# -lt 6 ]]; then
	echo "Usage: ./run.sh <mark> <data_file> <bfs_tile_size> <bfs_stripe_length> <pr_tile_size> <pr_stripe_length>"
	exit
fi

apps_dir=/home/zpeng/benchmarks/test
mark=$1
data_file=$2
bfs_tile_size=$3
bfs_stripe_length=$4
pr_tile_size=$5
pr_stripe_length=$6

output=output_$(date +%Y%m%d-%H%M%S)_graphPhi_${mark}.txt
:> $output

set -x

# BFS
echo "BFS:" >> $output
${apps_dir}/bfs_simd/bfs $data_file $bfs_tile_size $bfs_stripe_length >> $output
echo "" >> $output
#
## PageRank
#echo "PageRank:" >> $output
#${apps_dir}/pageRank_simd/page_rank $data_file $pr_tile_size $pr_stripe_length >> $output
#echo "" >> $output

## SSSP
#echo "SSSP:" >> $output
#${apps_dir}/sssp_simd/sssp ${data_file} $bfs_tile_size $bfs_stripe_length -w >> $output
#echo "" >> $output

## CC
#echo "CC:" >> $output
#${apps_dir}/connectedComponent_simd/cc $data_file $bfs_tile_size $bfs_stripe_length >> $output
#echo "" >> $output
#
## BC
#echo "BC:" >> $output
#${apps_dir}/betweennessCentrality_simd/bc ${data_file} $bfs_tile_size $bfs_stripe_length >> $output
#echo "" >> $output
#
## MIS
#echo "MIS:" >> $output
#${apps_dir}/maximalIndependentSet_simd/mis $data_file $bfs_tile_size $bfs_stripe_length >> $output
#echo "" >> $output

set +x
