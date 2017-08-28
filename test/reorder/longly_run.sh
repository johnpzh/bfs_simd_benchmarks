#!/usr/bin/bash

longly_do () {
	./page_rank ~/benchmarks/data/pokec/soc-pokec
	./page_rank ~/benchmarks/data/twt/out.twitter
	cd ../coo_tiling/
	./page_rank ~/benchmarks/data/pokec/soc-pokec 1024
	./page_rank ~/benchmarks/data/twt/out.twitter 4096
	cd ../coo_tiled_naive_bfs/
	make clean && make
	./bfs ~/benchmarks/data/pokec/soc-pokec 1024 > output_pokec.txt
	./bfs ~/benchmarks/data/twt/out.twitter 4096 > output_twt.txt
	cd ../coo_tiled_simd_bfs/
	make clean && make
	./bfs ~/benchmarks/data/pokec/soc-pokec 1024 > output_pokec.txt
	./bfs ~/benchmarks/data/twt/out.twitter 4096 > output_twt.txt
}
while true; do
	sw=$(top -b -n 1 -u rtian)
	if ! [[ "$sw" =~ .*rtian.* ]]
	then
		echo "Working starts."
		longly_do
		break
	fi
	sleep 1200
done
echo "Done."
