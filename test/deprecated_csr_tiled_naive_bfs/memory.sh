#!/usr/bin/bash
make

data_addr="/home/zpeng/benchmarks/data/pokec"
if [[ "$1" = "" ]]; then
	data_file="soc-pokec"
else
	data_file=$1
fi
version=naive-bfs-ddr
report_dir="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -quiet -- numactl -m 0 ./bfs ${data_addr}/${data_file} 1024
amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/write.txt
mv timeline.txt ${report_dir}/

#version=naive-bfs-mcdram
#report_dir="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
#amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -quiet -- numactl -m 1 ./bfs ${data_addr}/${data_file} 1024
#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/write.txt
#mv timeline.txt ${report_dir}/
echo done.
