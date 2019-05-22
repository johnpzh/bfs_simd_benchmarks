#! /usr/bin/bash
if [[ $# -lt 1 ]]; then
	echo "Usage: ./sh.sh <output_file>"
	exit
fi
fout=$1
app_dir=/home/zpeng/graphPhi/apps
:> $fout

# Naive
echo "Naive:" >> $fout
${app_dir}/bfs_naive/bfs /home/zpeng/data/twt/out.twitter_reorder 16384 1024 >> $fout
# Serial
echo "Serial:" >> $fout
${app_dir}/bfs_serial/bfs /home/zpeng/data/twt/out.twitter_reorder 16384 1024 >> $fout
# SIMD
echo "SIMD:" >> $fout
${app_dir}/bfs_simd/bfs /home/zpeng/data/twt/out.twitter 16384 1024 >> $fout
