#!/bin/bash
rm $1
while true;
do
        # free: Display memory details. -m to show values in mibibytes.
        # grep: search for 'Mem'
        # awk: print the value of the third column "used memory"
        nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader >> $1
        sleep 0.001; #second
#       echo "hi"
done