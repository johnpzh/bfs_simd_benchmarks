#! /usr/bin/bash

if [[ $# -lt 1 ]]; then
	echo "Usage: ./test_run.sh <data_dir>"
	exit
fi

dir=$1

# Pokec
./non-reorder.sh ${dir}/pokec/soc-pokec 8192 128
./non-reorder.sh ${dir}/pokec/soc-pokec_weighted 8192 128 weighted=1
./non-reorder.sh ${dir}/pokec/soc-pokec 4096 16

# Livejournal
./non-reorder.sh ${dir}/livejournal/livejournal 16384 256
./non-reorder.sh ${dir}/livejournal/livejournal_weighted 16384 256 weighted=1

# rmat24
./non-reorder.sh ${dir}/rmat24/rmat24 16384 128
./non-reorder.sh ${dir}/rmat24/rmat24_weighted 16384 128 weighted=1
./non-reorder.sh ${dir}/rmat24/rmat24 32768 512

# rmat27
./non-reorder.sh ${dir}/rmat27/rmat27 16384 256
./non-reorder.sh ${dir}/rmat27/rmat27_weighted 16384 256 weighted=1
./non-reorder.sh ${dir}/rmat27/rmat27 16384 8192

# Twt
./non-reorder.sh ${dir}/twt/out.twitter 16384 1024
./non-reorder.sh ${dir}/twt/out.twitter_weighted 16384 1024 weighted=1
./non-reorder.sh ${dir}/twt/out.twitter 32768 1024

# Friendster
./non-reorder.sh ${dir}/friendster/friendster 65536 512
./non-reorder.sh ${dir}/friendster/friendster_weighted 65536 512 weighted=1
./non-reorder.sh ${dir}/friendster/friendster 131072 512
