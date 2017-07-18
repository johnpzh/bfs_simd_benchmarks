#!/usr/bin/bash

longly_do () {
	./run.sh pokec
	./page_rank ~/benchmarks/data/soc-pokec-relationships.txt 1 128
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
