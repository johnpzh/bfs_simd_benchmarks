#!/usr/bin/bash
if [[ $# -lt 3 ]]; then
	echo "Usage: ./process.sh <data_file> <tile_size> <stripe_length>"
	exit
fi

data_file=$1
tile_size=$2
stripe_length=$3
# CSR tiling
cd /home/zpeng/benchmarks/test/csr_tiling
make clean
make untile=1
./page_rank $data_file 
# COO tiling

# Column-major

# CSR Tiling reverse

# COO Tiling reverse

# Column-major




#make
## Determine the data file
#if [[ $# -eq 0 ]]; then
#	data_file="soc-pokec-relationships.txt"
#else
#	case $1 in
#	pokec)
#		data_file="soc-pokec-relationships.txt"
#		;;
#	twt)
#		data_file="out.twitter_mpi"
#		;;
#	*)
#		data_file=$1
#		;;
#	esac
#fi
#bin_addr="."
#data_addr="/home/zpeng/benchmarks/data"
#
#power=10
#power_max=8
#size=$((2 ** power))
#while [	$power -le $power_max ]
#do
#	(set -x; ${bin_addr}/page_rank ${data_addr}/${data_file} ${size} 1024 2048 >> $result_file)
#	power=$((power + 1))
#	size=$((2 ** power))
#done
#echo done.
