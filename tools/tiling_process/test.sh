#! /usr/bin/bash

echo "plain degreeReordered"
./non-reorder.sh /data/zpeng/reorder_for_YuChen/twitter/out.twitter_degreeReordered 16384 1024

echo "reorder_degreeReordered"
./non-reorder.sh /data/zpeng/reorder_for_YuChen/twitter/out.twitter_reorder_degreeReordered 16384 1024

echo "weighted_reorder_degreeReordered"
./non-reorder.sh /data/zpeng/reorder_for_YuChen/twitter/out.twitter_weighted_reorder_degreeReordered 16384 1024 weighted=1
