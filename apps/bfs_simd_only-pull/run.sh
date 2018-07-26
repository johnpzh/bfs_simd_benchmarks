#! /usr/bin/bash
output="/home/zpeng/benchmarks/test/bfs_simd_only-pull/output_necessary_041118-134731.txt"
:> $output
make
set -x
# Only-Pull
echo "Only-Pull Pokec:" >> $output
./bfs /data/zpeng/pokec/soc-pokec 8192 128 >> $output
echo "Only-Pull Twt:" >> $output
./bfs /data/zpeng/twt/out.twitter 16384 1024 >> $output

# Pull/Push
cd /home/zpeng/benchmarks/test/bfs_simd_necessary-access
make
echo "Pull/Push Pokec:" >> $output
./bfs /data/zpeng/pokec/soc-pokec 8192 128 >> $output
echo "Pull/Push Twt:" >> $output
./bfs /data/zpeng/twt/out.twitter 16384 1024 >> $output

echo "Reorder+ Pokec:" >> $output
./bfs /data/zpeng/pokec/soc-pokec_reorder 8192 128 >> $output
echo "Reorder+ Twt:" >> $output
./bfs /data/zpeng/twt/out.twitter_reorder 16384 1024 >> $output


cd /home/zpeng/benchmarks/test/betweennessCentrality_simd
./bc /data/zpeng/rmat27/rmat27 16384 256 > "output_rmat27_dram.txt"

set +x
