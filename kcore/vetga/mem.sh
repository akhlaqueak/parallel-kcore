#!/bin/bash
while true;
do
        nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader
        sleep 0.01; 
done