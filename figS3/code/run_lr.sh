
#!/bin/bash

lrs='0.05 0.1 0.5 1.0'
seeds='30 43 56 77 123 78 57 23 167 333 45 27 893 456 73 55 234 536 230 39 3'


for i in $seeds:
do 
for j in $lrs:
do
python3 recurrent43.py $j $i
done
done