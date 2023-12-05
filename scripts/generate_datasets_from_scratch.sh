# Call this script to start the generation of the variants and the creation of
# the datasets

# download the wasms files
URL=$1
echo "Downloading wasms from $URL"
if [ ! -d wasms ]
then
    wget -O wasms.zip # https://zenodo.org/record/5832621/files/wasms.zip
    unzip wasms.zip
fi
# download and install wasm-mutate
if [ ! -d wasm-tools ]
then
    git clone https://github.com/bytecodealliance/wasm-tools.git
    cd wasm-tools
    cargo build --release
    cd ..
fi

dir_size=10 # change this variable to have more or less concurrent mutational processes
dir_name1="wasms/benign"
dir_name2="wasms/malign"

# Create the output folder, copy it to the notebooks root folder once this
# process is finished. Ensure you save the files inside the datasets folder,
# otherwise they will be overwritten
mkdir -p datasets

# Create the original dataset
python3 data_set_builder.py wasms/malign_all_hokon_april datasets/original_malign MALIGN
python3 pandarize.py datasets/original_malign datasets/original_malign.csv
python3 data_set_builder.py wasms/benign_all_hokon datasets/original_benign BENIGN
python3 pandarize.py datasets/original_benign datasets/original_benign.csv

exit
# Copy all wasms to the total folder
dir_name="total"


mkdir -p $dir_name
find $dir_name1 -name "*.wasm" -exec bash -c 'cp {} total/benign_$(basename {})' \;
find $dir_name2 -name "*.wasm" -exec bash -c 'cp {} total/malign_$(basename {})' \;

N=$((`find $dir_name -maxdepth 1 -type f -name "*.wasm" | wc -l`))
n=$((`find $dir_name -maxdepth 1 -type f -name "*.wasm" | wc -l`/$dir_size+1))

# Change the SEED to create another augmented distribution
SEED=0

#b#ash mutate.sh $SEED "total2/bening_www.qt.io.wasm"
#exit 1
echo "Binaries $N"
for i in `seq 1 $n`;
do
    rm -rf "total$i";
    mkdir -p "total$i";
    find $dir_name -maxdepth 1 -type f -name "*.wasm" | head -n $dir_size | xargs -I{} mv "{}" "total$i"

    # Execute wasm-mutate

    for w in total$i/*.wasm
    do
        bash mutate.sh $SEED "$w"  &
    done
    wait
done

rm -rf total*

# Create the datasets
python3 data_set_builder.py grouping/benign datasets/benign_augmented BENIGN
python3 pandarize.py datasets/benign_augmented datasets/benign_augmented.csv
python3 data_set_builder.py grouping/malign datasets/malign_augmented MALIGN
python3 pandarize.py datasets/malign_augmented datasets/malign_augmented.csv
