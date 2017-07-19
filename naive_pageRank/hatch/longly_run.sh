#!/usr/bin/bash

longly_do () {
	data_file=local_graph256MD4
	cd /home/zpeng/benchmarks/test/multi-io_ins_exe_vec/hatch
	./run.sh $data_file
	numactl -m 1 ./run.sh $data_file
	cd /home/zpeng/benchmarks/test/multi-io_ins_exe_vec/vtune
	./ddr_amplxe.sh $data_file
	./mcdr_amplxe.sh $data_file
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
