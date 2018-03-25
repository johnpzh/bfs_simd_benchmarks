#!/usr/bin/bash
make clean 
make debug=1
#make

#report_dir="report_m0_$(date +%Y%m%d-%H%M%S)"
#data_file="/home/zpeng/benchmarks/data/twt_combine/out.twitter"
#collect_flags="-collect concurrency -knob enable-user-tasks=true -knob enable-user-sync=true -knob analyze-openmp=true -data-limit=0 -result-dir ${report_dir}"
#summary_flags="-report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt"
#hotspot_flags="-report hotspots -result-dir ${report_dir} -format text -report-output ${report_dir}/hotspots.txt"
#
#amplxe-cl ${collect_flags} -- ./page_rank ${data_file} 4096 256
#amplxe-cl ${summary_flags}
#amplxe-cl ${hotspot_flags}

data_file="/home/zpeng/benchmarks/data/twt_combine/out.twitter"
for ((sl = 64; sl < 65; sl *= 2)); do
#for ((sl = 1; sl < 8193; sl *= 2)); do
	report_dir="report_$(date +%Y%m%d-%H%M%S)_sl${sl}_con"
	collect_flags="-collect concurrency -knob enable-user-tasks=true -knob enable-user-sync=true -knob analyze-openmp=true -data-limit=0 -result-dir ${report_dir}"
	summary_flags="-report summary -result-dir ${report_dir} -format text -report-output ${report_dir}/report.txt"
	hotspot_flags="-report hotspots -result-dir ${report_dir} -format text -report-output ${report_dir}/hotspots.txt"
	amplxe-cl ${collect_flags} -- ./page_rank ${data_file} 4096 ${sl}
	#amplxe-cl ${summary_flags}
	#amplxe-cl ${hotspot_flags}
done

