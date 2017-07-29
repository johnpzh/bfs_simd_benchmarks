#!/usr/bin/bash
make
version=naive-ddr
data_addr="/home/zpeng/benchmarks/data/twt"
if [[ "$1" = "" ]]; then
	data_file="out.twitter"
else
	data_file=$1
fi
report_dir_prefix="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
amplxe-cl -collect memory-access -result-dir ${report_dir_prefix} -data-limit=0 -quiet -- ./page_rank ${data_addr}/${data_file} 1
amplxe-cl -report summary -result-dir ${report_dir_prefix} -format text -report-output ${report_dir_prefix}/write.txt
mv timeline.txt ${report_dir_prefix}/
echo done.
