fn="memlog"
rm $fn
while true;
do
    nvidia-smi | grep 'python3' | awk '{print $8}' | grep -o -E '[0-9]+' >> $fn
    sleep 0.01
done