#!/usr/bin/bash

longly_do () {
	./run.sh
	./run.sh
	./memory.sh
	./memory.sh
	cd ../multiinput_naive_pageRank/
	./run.sh
	./run.sh
	./memory.sh
	./memory.sh
	cd ../simd_bfs/
	./run.sh
	./run.sh
	./memory.sh
	./memory.sh
	cd ../naive_bfs/
	./run.sh
	./run.sh
	./memory.sh
	./memory.sh
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
