#! /usr/bin/bash
galois_dir="/sciclone/home2/zpeng01/code/galois/build/release/tools/graph-convert-standalone"
tools_dir="/sciclone/home2/zpeng01/benchmarks/tools"
set -x

## Pokec
#./process.sh $galois_dir $tools_dir /data/zpeng/pokec/soc-pokec 1632803 30622564 
#
## livejournal
#./process.sh $galois_dir $tools_dir /data/zpeng/livejournal/livejournal 4847571 68475391
#
## rmat24
#./process.sh $galois_dir $tools_dir /data/zpeng/rmat24/rmat24 16777216 268435456

## usa_road
#./process.sh $galois_dir $tools_dir /data/zpeng/road_usa/road_usa 23947347 28854312

# Twt
#./process.sh $galois_dir $tools_dir /data/zpeng/twt/out.twitter 41652230 1468365182

## rmat27
#./process.sh $galois_dir $tools_dir /data/zpeng/rmat27/rmat27 134217728 2147483648

# Friendster
./process.sh $galois_dir $tools_dir /sciclone/scr-mlt/zpeng01/friendster/tmp_hima/friendster 68349466 2147483645


set +x
