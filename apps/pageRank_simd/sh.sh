#! /usr/bin/bash
if [[ $# -lt 1 ]]; then
	echo "Usage: ./sh.sh <output_file>"
	exit
fi
fout=$1
app_dir=/home/zpeng/graphPhi/apps
:> $fout

# Naive
echo "Naive:" >> $fout
${app_dir}/pageRank_naive/page_rank /home/zpeng/data/twt/out.twitter_reorder 32768 1024 2>&1 | tee -a $fout
# Serial
echo "Serial:" >> $fout
${app_dir}/pageRank_serial/page_rank /home/zpeng/data/twt/out.twitter_reorder 32768 1024 2>&1 | tee -a $fout
# SIMD
echo "SIMD:" >> $fout
${app_dir}/pageRank_simd/page_rank /home/zpeng/data/twt/out.twitter_reorder 32768 1024 2>&1 | tee -a $fout
