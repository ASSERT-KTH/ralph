use std::path::Path;
use std::fs;
use std::fs::File;
use flate2::read::GzDecoder;
use tar::Archive;

fn main() {
    let downloader = match std::env::consts::OS {
        "linux" => "curl",
        "macos" => "curl",
        "windows" => "wget",
        _ => panic!("Unsupported OS"),
    };
    let absolute_path = std::env::current_dir().unwrap();
    let absolute_path = format!("{absolute_path}/src", absolute_path=absolute_path.display());

    
    // Set environment variables
    // _malloc is needed to save the img data into the Wasm binary.
    println!("cargo:rustc-env=EMCC_CFLAGS=-sERROR_ON_UNDEFINED_SYMBOLS=0 -s TOTAL_STACK=32MB -s ASSERTIONS=2 -s TOTAL_MEMORY=256MB -s ALLOW_MEMORY_GROWTH=1 -sEXPORTED_FUNCTIONS=\"['_malloc', '_infer', '_init_model', '_print_metadata']\" --minify 0 -O0 -sMODULARIZE=1 -o dist/model.mjs ");
   

}