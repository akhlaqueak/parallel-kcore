res=(
Ours	"Ours +  SM"	"Ours + VP"	"Ballot scan"	"Ballot scan + SM"	"Ballot scan + 
VP"	"Efficient scan"	"Efficient scan + SM"	"Efficient Scan + VP"
)

for ds in "${res[@]}"; do
    echo "$ds"
    grep "$ds" log/publish.out | awk '{print $NF}'
done