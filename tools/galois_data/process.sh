#! /usr/bin/bash
if [[ $# -lt 5 ]]; then
	echo "Usage: ./process.sh <graph-convert_dir> <tools_dir> <data_file> <num of vertices> <num of edges>"
	exit
fi
galois_dir=$1
tools_dir=$2
data_file=$3
nnodes=$4
nedges=$5

nohead_data=${data_file}_nohead
weighted_data=${data_file}_weighted

set -x

## Vertex id subtract 1 and delete the first line
#${tools_dir}/vertex_id_sub_one/page_rank $data_file
#
# Edge list -> binary void gr
${galois_dir}/graph-convert-standalone ${nohead_data} ${data_file}.vgr -edgelist2vgr
echo "Got ${data_file}.vgr"

# Binary void gr -> binary weighted gr
${galois_dir}/graph-convert-standalone ${data_file}.vgr ${data_file}.gr -vgr2intgr
echo "Got ${data_file}.gr"

# Binray void gr transpose
${galois_dir}/graph-convert-standalone ${data_file}.vgr ${data_file}.tvgr -vgr2tvgr
echo "Got ${data_file}.gr"

# Binary weighted gr transpose
${galois_dir}/graph-convert-standalone ${data_file}.gr ${data_file}.tgr -gr2tintgr
echo "Got ${data_file}.tgr"

## Galois format -> Ligra (pbbs) format
#${galois_dir}/graph-convert-standalone ${data_file}.vgr ${data_file}.adj -vgr2pbbs
#echo "Got ${data_file}.adj"
#${galois_dir}/graph-convert-standalone ${data_file}.gr ${weighted_data}.adj -gr2intpbbs
#echo "Got ${weighted_data}.adj"
#
## Binary Weighted gr -> weighted Edge List
#${galois_dir}/graph-convert-standalone ${data_file}.gr ${weighted_data} -gr2intpbbsedges
#${tools_dir}/vertex_id_add_one/page_rank ${weighted_data} $nnodes $nedges
#mv ${weighted_data} ${weighted_data}.pdds
#mv ${weighted_data}_weighted ${weighted_data}
#echo "Got ${weighted_data}"

set +x
