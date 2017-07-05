#!/usr/bin/bash

longly_do () {
	cd /home/zpeng/benchmarks/test/multi-io_ins_exe_vec/vtune
	./ddr_amplxe.sh graph256MD4
	./mcdr_amplxe.sh graph256MD4
	cd /home/zpeng/benchmarks/test/origin_rodinia/vtune
	./ddr_amplxe.sh graph256MD4
	./mcdr_amplxe.sh graph256MD4
	cd /home/zpeng/benchmarks/test/rodinia_simd/vtune
	./ddr_amplxe.sh graph256MD4
	./mcdr_amplxe.sh graph256MD4
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
