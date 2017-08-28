#!/usr/bin/bash
path2=/home/zpeng/benchmarks/test/tw_naive_bfs/path
for ((i = 0; i < 64; ++i)); do
	diff path/path${i}.txt ${path2}/path${i}.txt
done
