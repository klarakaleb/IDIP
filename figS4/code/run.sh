#!/bin/bash

seeds='30 43 56 77 123 78 57 23 167 333 45 27 893 456 73 55 234 536 230 39 3'

for i in $seeds:
do
python3 recurrent.py $i
done