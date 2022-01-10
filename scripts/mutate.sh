
ITERATIONS=1000
RANDOM=$1
WASM_MUTATE="./wasm-tools/target/release/wasm-tools mutate"

mkdir -p grouping/malign
mkdir -p grouping/benign

ARGS="--preserve-semantics"
GENERATION_TIMEOUT=10

function mutate() {
    echo "Mutating " $1
    if [[ $(basename $1) == "malign"* ]]
    then
        OUT="grouping/malign"
    else
        OUT="grouping/benign"
    fi

    LAST=$1
    i=0
    TRIES=1000
    while [ $i -lt $2 ]
    do
        if [ $TRIES == 0 ]
        then
            break
        fi

        # check prefix
        timeout $GENERATION_TIMEOUT $WASM_MUTATE $LAST $ARGS --seed $RANDOM -o $OUT/$(basename $1).$i.wasm 2> /dev/null
        code=$?
        if [[ $code == 0 ]]
        then
            LAST=$OUT/$(basename $1).$i.wasm            
            ((i=i+1))
            TRIES=1000
        else
            rm $OUT/$(basename $1).$i.wasm 2> /dev/null
            ((TRIES=TRIES-1))
        fi
    done
}


mutate $2 $ITERATIONS
