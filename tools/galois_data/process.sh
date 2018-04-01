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

# Vertex id subtract 1 and delete the first line
#/home/zpeng/benchmarks/tools/vertex_id_sub_one/page_rank $data_file
${tools_dir}/vertex_id_sub_one/page_rank $data_file

# Edge list -> binary void gr
#/home/zpeng/code/galois_set/build/release/tools/graph-convert/graph-convert ${nohead_data} ${data_file}.vgr -edgelist2vgr
${galois_dir}/graph-convert ${nohead_data} ${data_file}.vgr -edgelist2vgr
#rm $nohead_data
echo "Got ${data_file}.vgr"

# Binary void gr -> binary weighted gr
#/home/zpeng/code/galois_set/build/release/tools/graph-convert/graph-convert ${data_file}.vgr ${data_file}.gr -vgr2intgr
${galois_dir}/graph-convert ${data_file}.vgr ${data_file}.gr -vgr2intgr
echo "Got ${data_file}.gr"

# Binary weighted gr transpose
#/home/zpeng/code/galois_set/build/release/tools/graph-convert/graph-convert ${data_file}.gr ${data_file}.tgr -vgr2intgr
${galois_dir}/graph-convert ${data_file}.gr ${data_file}.tgr -vgr2intgr
echo "Got ${data_file}.tgr"

# Galois format -> Ligra (pbbs) format
#/home/zpeng/code/galois_set/build/release/tools/graph-convert/graph-convert ${data_file}.vgr ${data_file}.adj -vgr2pbbs
${galois_dir}/graph-convert ${data_file}.vgr ${data_file}.adj -vgr2pbbs
echo "Got ${data_file}.adj"
#/home/zpeng/code/galois_set/build/release/tools/graph-convert/graph-convert ${data_file}.gr ${weighted_data}.adj -gr2intpbbs
${galois_dir}/graph-convert ${data_file}.gr ${weighted_data}.adj -gr2intpbbs
echo "Got ${weighted_data}.adj"

# Binary Weighted gr -> weighted Edge List
#/home/zpeng/code/galois_set/build/release/tools/graph-convert/graph-convert ${data_file}.gr ${weighted_data} -gr2intpbbsedges
${galois_dir}/graph-convert ${data_file}.gr ${weighted_data} -gr2intpbbsedges
#/home/zpeng/benchmarks/tools/vertex_id_add_one/page_rank ${weighted_data} $nnodes $nedges
${tools_dir}/vertex_id_add_one/page_rank ${weighted_data} $nnodes $nedges
#rm ${weighted_data}
mv ${weighted_data} ${weighted_data}.bak
mv ${weighted_data}_weighted ${weighted_data}
echo "Got ${weighted_data}"
