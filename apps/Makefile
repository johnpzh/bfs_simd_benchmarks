
DIR = bfs pageRank sssp cc bc mis

all: $(DIR)

bfs:
	cd bfs_simd && $(MAKE)

pageRank:
	cd pageRank_simd && $(MAKE)

sssp:
	cd sssp_simd && $(MAKE)

cc:
	cd connectedComponent_simd && $(MAKE)

bc:
	cd betweennessCentrality_simd && $(MAKE)

mis:
	cd maximalIndependentSet_simd && $(MAKE)
