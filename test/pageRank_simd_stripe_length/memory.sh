#!/usr/bin/bash
make clean
make debug=1

data_file="/data/zpeng/twt_combine/tiled_2-power/out.twitter"
# Tile Size
for ((i = 1024; i < 524289; i *= 2)); do
	report_dir="report_$(date +%Y%m%d-%H%M%S)_tile-size-${i}_memory"
	amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- numactl -m 0 ./page_rank ${data_file} ${i} 40
	amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
done

# Stripe Length
for ((i = 1; i < 513; i *= 2)); do
	report_dir="report_$(date +%Y%m%d-%H%M%S)_stripe-length-${i}_memory"
	amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- numactl -m 0 ./page_rank ${data_file} 65536 ${sl}
	amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
done
