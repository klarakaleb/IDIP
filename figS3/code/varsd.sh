#!/bin/bash

cs='0.0 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.45 0.475 0.5 0.5'
seeds='30 43 56 77 123 78 57 23 167 333 45 27 893 456 73 55 234 536 230 39 3'


for i in $cs:
do
for j in $seeds:
do
python3 recurrent_varsd.py $i $j
done
done

