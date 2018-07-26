#!/usr/bin/bash
if [[ $# -lt 5 ]]; then
	echo "Usage: ./memory.sh <mark> <data_file> <tile_size> <min_stripe_length> <max_stripe_length>"
	exit
fi

mark=$1
data_file=$2
tile_size=$3
min_stripe_length=$4
max_stripe_length=$5
## Tile Size
#for ((i = 1024; i < 524289; i *= 2)); do
#	report_dir="report_$(date +%Y%m%d-%H%M%S)_tile-size-${i}_memory"
#	amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- numactl -m 0 ./page_rank ${data_file} ${i} 40
#	#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
#done

make clean
make debug=1
#set -x
# Stripe Length
for ((i = min_stripe_length; i <= max_stripe_length; i *= 2)); do
	#report_dir="report_$(date +%Y%m%d-%H%M%S)_stripe-length-${i}_memory_${mark}"
	#amplxe-cl -collect memory-access -result-dir ${report_dir} -data-limit=0 -knob analyze-mem-objects=true -knob analyze-openmp=true -- ./page_rank ${data_file} ${tile_size} ${i}
	#amplxe-cl -report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt
	./page_rank ${data_file} ${tile_size} ${i}
done

#set +x
