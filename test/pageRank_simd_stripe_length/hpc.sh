#!/usr/bin/bash
make clean 
make debug=1
#make

## Stripe Length
#data_file="/data/zpeng/twt_combine/tiled_2-power/out.twitter"
#for ((sl = 1; sl < 513; sl *= 2)); do
#	report_dir="report_$(date +%Y%m%d-%H%M%S)_stripe-length-${sl}_hpc"
#	amplxe-cl -collect hpc-performance -result-dir ${report_dir} -data-limit=0 -- numactl -m 0 ./page_rank ${data_file} 65536 ${sl}
#	#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
#done

# Tile Size
for ((i = 1024; i < 524289; i *= 2)); do
	report_dir="report_$(date +%Y%m%d-%H%M%S)_tile-size-${i}_hpc"
	amplxe-cl -collect hpc-performance -result-dir ${report_dir} -data-limit=0 -- numactl -m 0 ./page_rank ${data_file} ${i} 40
	#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
done
