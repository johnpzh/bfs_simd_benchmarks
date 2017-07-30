#!/usr/bin/bash
make

data_addr="/home/zpeng/benchmarks/data/rodinia_gen"
if [[ "$1" = "" ]]; then
	data_file="graph256MD4"
else
	data_file=$1
fi
version=simd-bfs-vertices
report_dir="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -quiet -- ./bfs 1 ${data_addr}/${data_file} 256 32768
amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/write.txt
mv timeline.txt ${report_dir}/

echo done.
