#!/usr/bin/bash

for ((i = 1500; i < 6501; i += 500)); do
	(set -x; ./bfs /home/zpeng/benchmarks/data/twt_combine/tiled_1500/out.twitter $i 256)
done
