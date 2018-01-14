#!/usr/bin/bash
make

#data_addr="/home/zpeng/benchmarks/data/pokec"
#data_file="soc-pokec_nohead.adj"
data_addr="/home/zpeng/benchmarks/data/twt"
data_file="out.twitter_nohead.adj"
version=ligra-bfs
report_dir="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -quiet -- numactl -m 0 ./BFS ${data_addr}/${data_file}
amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/write.txt
mv timeline.txt ${report_dir}/

#version=naive-bfs-mcdram
#report_dir="report_${version}_${data_file}_$(date +%Y%m%d-%H%M%S)"
#amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -quiet -- numactl -m 1 ./bfs ${data_addr}/${data_file}
#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/write.txt
#mv timeline.txt ${report_dir}/
echo done.
