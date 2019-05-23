#! /usr/bin/bash

# Twitter
make clean
make
./bfs /data/zpeng/reorder_for_YuChen/twitter/out.twitter 

# Twitter reorder
./bfs /data/zpeng/reorder_for_YuChen/twitter/out.twitter_reorder

# Twitter weighted reorder
make clean
make weighted=1
./bfs /data/zpeng/reorder_for_YuChen/twitter/out.twitter_weighted_reorder

