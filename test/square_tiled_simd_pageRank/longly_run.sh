#!/usr/bin/bash

longly_do () {
	./run.sh
	./memory.sh
	cd ../multiinput_naive_pageRank/
	./run.sh
	./memory.sh
	cd ../simd_bfs/
	./run.sh
	./memory.sh
	cd ../naive_bfs/
	./run.sh
	./memory.sh
	cd ../edges_naive_pageRank/
	./longly_run.sh
}

longly_do
#while true; do
#	sw=$(w)
#	if ! [[ "$sw" =~ .*rtian.* ]]
#	then
#		echo "Working starts."
#		longly_do
#		break
#	fi
#	sleep 600
#done
echo "Done."
