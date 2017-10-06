#!/usr/bin/bash
data_file_origin="/home/zpeng/benchmarks/data/twt/out.twitter"
data_file_sym="/home/zpeng/benchmarks/data/twt_sym/out.twitter"
make clean && make debug=1
(set -x; ./kcore $data_file_origin > output_debug_origin.txt)
(set -x; ./kcore $data_file_sym > output_debug_sym.txt)

make clean && make
(set -x; ./kcore $data_file_origin > output_optim_origin.txt)
(set -x; ./kcore $data_file_sym > output_optim_sym.txt)
echo done.
