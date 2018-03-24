#!/usr/bin/bash
make clean
make debug=1

#data_addr="/home/zpeng/benchmarks/data/pokec/coo_tiled_bak"
#data_file="soc-pokec"
data_file="/home/zpeng/benchmarks/data/twt_combine/out.twitter"

for ((sl = 1; sl < 8193; sl *= 2)); do
	report_dir="report_$(date +%Y%m%d-%H%M%S)_sl${sl}_memory"
	amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- numactl -m 0 ./page_rank ${data_file} 4096 ${sl}
	amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
done

#version=simd-bfs-m1
#report_dir="report_${version}_$(date +%Y%m%d-%H%M%S)"
#amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- numactl -m 1 ./page_rank ${data_file} 4096 256
#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
#mv timeline.txt ${report_dir}/
#echo done.
#
#version=simd-bfs-p1
#report_dir="report_${version}_$(date +%Y%m%d-%H%M%S)"
#amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- numactl -p 1 ./page_rank ${data_file} 4096 256
#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
#mv timeline.txt ${report_dir}/
#echo done.
