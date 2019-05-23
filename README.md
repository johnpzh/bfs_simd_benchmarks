# Parallel Graph Processing
Main ideas:
1. Use tiling, reordering to process COO data.
2. Use Top-down, and Bottom-up processing pattern for frontier-kind algorithm (e.g. BFS).
3. Use OpenMP for MIMD, and AVX-512 for SIMD.
4. Run on Intel Knights Landing machine with MCDRAM.

For more details please refer to the paper
> [GraphPhi: Efficient Parallel Graph Processing on Emerging Throughput-oriented Architectures (PACT 2018)](https://dl.acm.org/citation.cfm?id=3243205)

# Running the code:

1.  Clone the repository: `git clone https://github.com/johnpzh/bfs_simd_benchmarks.git`
2.  Dataset: 
	* Pokec: [http://konect.uni-koblenz.de/networks/soc-pokec-relationships](http://konect.uni-koblenz.de/networks/soc-pokec-relationships)
	* Twitter: [http://konect.uni-koblenz.de/networks/twitter](http://konect.uni-koblenz.de/networks/twitter)

3.  Download the dataset:
	1. `mkdir datafolder`
	2. `wget http://konect.uni-koblenz.de/downloads/tsv/soc-pokec-relationships.tar.bz2`
	3. Got the bz2 file
	4. Then decompress it: `tar jxvf soc-pokec-relationships.tar.bz2`
	5. We only need the “out.xxx” file. Delete the first two lines of the file (those lines starting with %).
	6. Insert at the very beginning of the file the number of vertices and number of edges as the first line, like “1632803 30622564”. Save and exit.

4. CSR tiling:   
	1.  `cd BENCHMARK/tools/csr_tiling`
	2. Compile for unweighted graph: `make`
	3.  Or compile for weighted graph: `make weighted=1`
	4.  Run: `./page_rank DATAFOLDER/out.xxx` (Then we got 64 files named out.xxx_untiled-xx)
    
5.  COO tiling:
	1.  `cd BENCHMARK/tools/coo_tiling`
	2.  Compile and link: `make`
	3.  Run: `./page_rank DATAFOLDER/out.xxx 1024 1024` (here it needs two 1024, then we got 64 files named out.xxx_coo-tiled-1024-xx; 1024 is the tile width, we may change it accordingly.)
    
6.  Column-major modifying:
	1.  `cd BENCHMARK/tools/column_major_tile`
	2.  Compile and link: `make`
	3.  Run: `./kcore DATAFOLDER/out.xxx 1024 16 16` (here it needs two 16, then we got 64 files named out.xxx_col-16-coo-tiled-1024-xx; 16 is the stripe length, we may change it accordingly.)

7.  Reorder (according to BFS accessing order)
	1.  Need those **64** xxx_untiled files as input.	    
	2.  `cd BENCHMARK/tools/vertex_id_remap`
	3.  Compile for unweighted graph: `make clean && make`
	4.  Or compile for weighted graph: `make clean && make weighted=1`
	5.  Run: `./bfs DATAFOLDER/xxx`
	6.  The output is one file in `DATAFOLDER/xxx_reorder`
    
8.  Reorder (based on vertex degrees, then vertex 0 has the most degrees)
	1.  Need those **64** xxx_untiled files as input.
	2.  `cd BECHMARK/tools/vertex_id_reorder_to_degrees`
	3.  Compile for unweighted graph: `make clean && make`
	4.  Or compile for weighted graph: `make clean && make weighted=1`
	5.  Run: `./bfs DATAFOLDER/xxx`    
	6.  The output is one file in `DATAFOLDER/xxx_degreeReordered`
    

9.  Weighted graph generation:
	1.  Use the program provided by Galois to generate a weighted graph from edge-list graph file.
	2.  Galois is hosted on GitHub now: [https://github.com/IntelligentSoftwareSystems/Galois](https://github.com/IntelligentSoftwareSystems/Galois), installation is needed.
	3.  The tools is in `<Galois_path>/tools/graph-convert`
    
10.  Then we can run the program.
     1. `cd BENCHMARK/test/bfs_serial`  
	 2. Compile and link: `make`
	 3.  Run: `./bfs DATAFOLDER/out.xxx 1024 16 (-w)`
	 4.  `# bfs is the benchmark, it could be pagerank or any others.`
	 5.  `# 1024 is tile width, 16 is stripe length, and they should consist with those which are generated in previous steps`
    
11.  If you want to run other versions, like “bfs_top-down_recur”, the steps are the same:
		1.  `cd …`
		2.  Check the arguments for the program
		3.  `make`  
		4.  `./bfs <arguments <, ...>>`
