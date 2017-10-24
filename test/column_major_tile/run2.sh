#!/usr/bin/bash
make clean && make
for ((i=1; i<3; ++i)); do
	row_step=$((2**i))
	(set -x; ./kcore ~/benchmarks/data/twt_col/out.twitter 4096 $row_step)
done
echo done.
