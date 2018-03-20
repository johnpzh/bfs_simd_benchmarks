#! /usr/bin/bash
data_file="/home/zpeng/benchmarks/data/twt_combine/out.twitter"
#data_file="/home/zpeng/benchmarks/data/pokec_combine/soc-pokec"

set -x

# BC
#cd /home/zpeng/benchmarks/test/betweennessCentrality_serial/
#./bc ${data_file} 4096 256 > output.txt

# PageRank stripe length bandwidth
#cd ../pageRank_simd_stripe_length/
#./memory.sh

# PageRank buffer size
cd /home/zpeng/benchmarks/test/pageRank_simd_buffer_size/
numactl -m 0 ./page_rank ${data_file} 4096 40 > output_m0.txt
numactl -p 1 ./page_rank ${data_file} 4096 40 > output_p1.txt

# BC
cd /home/zpeng/benchmarks/test/betweennessCentrality_serial/
./bc /home/zpeng/benchmarks/data/twt_reorder/out.twitter_reorder 4096 256 > output.txt

set +x
