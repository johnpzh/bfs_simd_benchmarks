#! /usr/bin/bash

set -x

#./graphPhi_run.sh pokec /data/zpeng/pokec/soc-pokec 8192 128 4096 16
#./graphPhi_run.sh livejournal /data/zpeng/livejournal/livejournal 16384 256 16384 256
#./graphPhi_run.sh rmat24 /data/zpeng/rmat24/rmat24 16384 128 32768 512
##./graphPhi_run.sh road_usa_sssp /data/zpeng/road_usa/road_usa 16384 512 262144 64
./graphPhi_run.sh twt /data/zpeng/twt/out.twitter 16384 1024 32768 1024
#./graphPhi_run.sh rmat27 ~/data/rmat27/rmat27 16384 256 16384 8192
#./graphPhi_run.sh friendster /data/zpeng/friendster/friendster 65536 512 131072 512
set +x
