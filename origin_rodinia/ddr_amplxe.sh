#!/usr/bin/bash

version=r-origin-ddr
data_addr="/home/zpeng/benchmarks/data"
if [[ "$1" = "" ]]; then
	data_file="graph256MD4"
else
	data_file=$1
fi
report_dir_prefix="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
power_max=9
for ((i = 0; i < power_max; ++i)); do
	size=$((2 ** i))
	report_dir=${report_dir_prefix}_t${size}
	amplxe-cl -collect memory-access -result-dir ${report_dir} -quiet -- ./bfs $size ${data_addr}/${data_file}
#amplxe-cl -collect general-exploration -knob dram-bandwidth-limits=true -knob collect-memory-bandwidth=true -result-dir ${report_dir} -quiet -- ./bfs $size ${data_addr}/${data_file} 256 32768
#amplxe-cl -collect hpc-performance -knob dram-bandwidth-limits=true -knob collect-memory-bandwidth=true -result-dir ${report_dir} -quiet -- ./bfs $size ${data_addr}/${data_file} 256 32768
	amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/write.txt
	echo -n .
done
echo done.
