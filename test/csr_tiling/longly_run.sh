#!/usr/bin/bash

longly_do () {
	make
	./page_rank ~/benchmarks/data/twt/out.twitter 4096 > output.txt
}
while true; do
	sw=$(top -n 1 -b -u rtian)
	if ! [[ "$sw" =~ .*rtian.* ]]
	then
		echo "Working starts."
		longly_do
		break
	fi
	sleep 1800
done
echo "Done."
