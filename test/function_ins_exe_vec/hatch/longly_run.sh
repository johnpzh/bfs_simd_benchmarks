#!/usr/bin/bash

while true
do
	sw=$(w)
	if ! [[ "$sw" =~ .*rtian.* ]]
	then
		echo "Working starts."
#./graphgen 16777216 16M_d128
		for ((i = 0; i < 4; ++i)); do
			./run.sh 16M
		done
		break
	fi
	sleep 600
done
echo "Done."
