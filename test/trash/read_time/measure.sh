#!/usr/bin/bash

read_threads=1
max_threads=64
output=report.txt

touch $output

while [[ $read_threads -le $max_threads ]]; do
	echo "******* Threads: $read_threads *******" >> $output
	time ./bfs 64 i/home/zpeng/benchmarks/rodinia_3.1/data/bfs/graph16M.bak.txt 65536 4096 $read_threads >> $output
	read_threads=$((read_threads + 1))
done
