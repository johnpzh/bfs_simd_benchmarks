#!/usr/bin/bash

longly_do () {
	fout="output.txt"
	./run.sh >> $fout
	./memory.sh >> $fout
	cd /home/zpeng/benchmarks/test/vertices_naive_pageRank
	./run.sh >> $fout
	./memory.sh >> $fout
	cd /home/zpeng/benchmarks/test/edges_simd_pageRank
	./run.sh >> $fout
	./memory.sh >> $fout
	cd /home/zpeng/benchmarks/test/vertices_simd_pagaRank
	./run.sh >> $fout
	./memory.sh >> $fout
	cd /home/zpeng/benchmarks/test/edges_naive_bfs
	./run.sh >> $fout
	./memory.sh >> $fout
	cd /home/zpeng/benchmarks/test/vertices_naive_bfs
	./run.sh >> $fout
	./memory.sh >> $fout
	cd /home/zpeng/benchmarks/test/edges_simd_bfs
	./run.sh >> $fout
	./memory.sh >> $fout
	cd /home/zpeng/benchmarks/test/vertices_simd_bfs
	./run.sh >> $fout
	./memory.sh >> $fout
}

while true; do
	sw=$(users)
	if ! [[ "$sw" =~ .*rtian.* ]]
	then
		echo "Working starts."
		longly_do
		break
	fi
	sleep 600
done
echo "Done."
