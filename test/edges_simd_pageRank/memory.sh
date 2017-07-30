#!/usr/bin/bash
make

data_addr="/home/zpeng/benchmarks/data/twt"
if [[ "$1" = "" ]]; then
	data_file="out.twitter"
else
	data_file=$1
fi
version=simd-pageRank-edges
report_dir="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -quiet -- ./page_rank ${data_addr}/${data_file} 1 4096 512
amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/write.txt
mv timeline.txt ${report_dir}/

echo done.
