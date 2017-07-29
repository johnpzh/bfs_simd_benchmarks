#!/usr/bin/bash

longly_do () {
	data_file="~/benchmarks/data/out.twitter_mpi"
	tile_width=4096
	make 
	./page_rank $data_file $tile_width
	cd /home/zpeng/benchmarks/test/square_tiled_simd_pageRank
	./page_rank $data_file $tile_width 4096
}
while true; do
	sw=$(w)
	if ! [[ "$sw" =~ .*rtian.* ]]
	then
		echo "Working starts."
		longly_do
		break
	fi
	sleep 600
done
echo "Done."
