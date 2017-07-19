#!/usr/bin/bash

longly_do () {
	data_file=pokec
	./run.sh $data_file
	cd /home/zpeng/benchmarks/test/tiled_single_pageRank/hatch
	./run.sh $data_file
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
