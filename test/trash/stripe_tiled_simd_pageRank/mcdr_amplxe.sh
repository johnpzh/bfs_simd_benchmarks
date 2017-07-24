#!/usr/bin/bash

version=stripe-simd-pageRank-mcdram
data_addr="/home/zpeng/benchmarks/data"
if [[ "$1" = "" ]]; then
	data_file="soc-pokec-relationships.txt"
else
	data_file=$1
fi
report_dir_prefix="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
#power_max=9
#for ((i = 0; i < power_max; ++i)); do
#	size=$((2 ** i))
#	report_dir=${report_dir_prefix}_t${size}
#	amplxe-cl -collect memory-access -result-dir ${report_dir} -quiet -- numactl -m 1 ./bfs $size ${data_addr}/${data_file} 256 32768
#	amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/write.txt
#	echo -n .
#done
amplxe-cl -collect memory-access -result-dir ${report_dir_prefix} -quiet -- numactl -m 1 ./page_rank ${data_addr}/${data_file} 1 32 > ${report_dir_prefix}_time-line.txt
amplxe-cl -report summary -result-dir ${report_dir_prefix} -format text -report-output ${report_dir}/write.txt
echo done.
