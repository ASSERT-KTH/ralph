# re-implementation of the Minos WebAssembly malware detector 


# Requirements

- Python 3.9.9

# Installation

- `virtualenv -p <python3.9.9> ralph`
- `source ralph/bin/activate`
- `pip install -r requirements.txt`

## Generate the datasets from sratch

- `source ralph/bin/activate`
- `cd scripts`
- `bash generate_datasets_from_scratch.sh "https://zenodo.org/record/5832621/files/wasms.zip"`

## Train the model

- Train and save the h5 model `python3 minos.py train -b datasets/original_benign.csv -m datasets/original_malign.csv --model model.h5`
- Train and save the h5 model and also the onnx model `python3 minos.py train -b datasets/original_benign.csv -m datasets/original_malign.csv --model model.onnx`

## Infer from a Wasm binary

- `python3 minos.py predict -i test.wasm   `


# Repo structure
- `.gihub`: Contains the CI jobs to train and save the models in the artifact storage
- `scripts`: MINOS implementation(`minos.py`), the scripts to turn Wasm binaries into 100x100 grayscale images, and the (TODO) wasm_wrapper to make browser client inference with the already trained model.
- `scripts/wasm_wrapper`: TODO 