#! /usr/bin/bash

pact_path="/home/zpeng/pact2018/apps/simd_performance"

cp betweennessCentrality_serial/bc.cpp ${pact_path}/bc/bc_serial.cpp
cp connectedComponent_serial/cc.cpp ${pact_path}/cc/cc_serial.cpp
cp maximalIndependentSet_serial/mis.cpp ${pact_path}/mis/mis_serial.cpp
cp pageRank_serial/page_rank.cpp ${pact_path}/pagerank/pagerank_serial.cpp
cp sssp_serial/sssp.cpp ${pact_path}/sssp/sssp_serial.cpp
