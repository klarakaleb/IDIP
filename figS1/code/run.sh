#!/bin/bash

seeds='30 122 229 515 976 128 669 845 142 40 722 93 144 423 772 969 64 682 198 333 333'
targets='10 15 20 25 30 30'

for i in $seeds:
do
for j in $targets:
do
python3 place_cells.py $i $j 
done
done