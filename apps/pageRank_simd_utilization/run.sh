#! /usr/bin/bash

set -x
# BUffer
cd /home/zpeng/benchmarks/test/pageRank_simd_utilization
./page_rank /data/zpeng/twt/out.twitter 4096 1024 > twt_4096-1024_output.txt

# No-buffer
cd /home/zpeng/benchmarks/test/pageRank_simd_no_buffer
./page_rank /data/zpeng/twt/out.twitter 4096 1024 > twt_4096-1024_output.txt


set +x
