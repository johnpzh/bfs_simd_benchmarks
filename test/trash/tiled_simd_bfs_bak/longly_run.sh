#!/usr/bin/bash

longly_do () {
	make
	./run.sh
	cd ../tw_naive_bfs/
	make
	./run.sh
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
