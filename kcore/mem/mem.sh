#!/bin/bash
while true;
do
        # free: Display memory details. -m to show values in mibibytes.
        # grep: search for 'Mem'
        # awk: print the value of the third column "used memory"
        nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader
        sleep 0.01; 
done


# fn="memlog"
# rm $fn
# while true;
# do
#     nvidia-smi | grep 'python3' | awk '{print $8}' | grep -o -E '[0-9]+' >> $fn
#     sleep $1
# done


# cd $(dirname $0) # Jalal: the folder containing the script will be the root folder. https://askubuntu.com/a/368100
# total_max=0
# f="memlog"

# max=`awk 'BEGIN{a=0}{if ($1>0+a) a=$1} END{print a}' $f`;
# min=0;  #`awk 'BEGIN{a=9999999}{if ($1<0+a) a=$1} END{print a}' $f`;

# dif="$(($max-$min))";
# if [ "$dif" -gt "$total_max" ]; then
#     total_max=$dif
# fi
#     #echo $dif;

#     # -f: force, "delete the file if exist, no error if doesn't"
# rm -f $f;
# echo "mem(MB): $total_max"
