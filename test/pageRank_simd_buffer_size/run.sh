#! /usr/bin/bash

data_file="/home/zpeng/benchmarks/data/twt_combine/out.twitter"
#data_file="/home/zpeng/benchmarks/data/pokec_combine/soc-pokec"
set -x
numactl -m 0 ./page_rank ${data_file} 4096 256 > output_m0.txt
numactl -p 1 ./page_rank ${data_file} 4096 256 > output_p1.txt
set +x
