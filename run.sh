#Set the place where result saved
outout_root=./data
fasta_path=$outout_root/amp.fasta
mkdir $outout_root

#Train model with example fasta
python3 train.py --f $fasta_path -o $outout_root --b 128 --s 10 --e 1000
