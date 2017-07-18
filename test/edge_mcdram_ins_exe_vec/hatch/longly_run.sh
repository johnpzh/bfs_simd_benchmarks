#!/usr/bin/bash

longly_do () {
	data_file=graph256MD4
	cd /home/zpeng/benchmarks/test/vetex_mcdram_ins_exe_vec/hatch
	for ((i = 0; i < 4; ++i)); do
		./run.sh $data_file
	done
	cd /home/zpeng/benchmarks/test/edge_mcdram_ins_exe_vec/hatch
	for ((i = 0; i < 4; ++i)); do
		./run.sh $data_file
	done
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
