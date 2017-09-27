#!/usr/bin/bash

longly_do () {
	./run.sh
}

while true; do
	sw=$(top -b -n 1 -u rtian)
	if ! [[ "$sw" =~ .*rtian.* ]]
	then
		echo "Working starts."
		longly_do
		break
	fi
	sleep 600
done
echo "Done."
