#!/usr/bin/bash
make

report_dir="report_m0_$(date +%Y%m%d-%H%M%S)"
data_file="/home/zpeng/benchmarks/data/twt_reorder/out.twitter_reorder"
collect_flags="-collect concurrency -knob enable-user-tasks=true -knob enable-user-sync=true -knob analyze-openmp=true -data-limit=0 -result-dir"
summary_flags="-report summary -result-dir ${report_dir} -format text -report-output"
hotspot_flags="-report hotspots -result-dir ${report_dir} -format text -report-output"

amplxe-cl ${collect_flags} ${report_dir} -- ./bfs ${data_file} 4096 256
amplxe-cl ${summary_flags} ${report_dir}/report.txt
amplxe-cl ${hotspot_flags} ${report_dir}/hotspots.txt

#version=simd-bfs-m0
#report_dir="report_${version}_$(date +%Y%m%d-%H%M%S)"
#amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- numactl -m 0 ./page_rank ${data_file} 4096 256
#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
#mv timeline.txt ${report_dir}/
#echo done.
#
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
