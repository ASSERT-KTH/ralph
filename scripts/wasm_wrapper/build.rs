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
    let absolute_path = format!("{absolute_path}", absolute_path=absolute_path.display());

    
    // Set environment variables
    println!("cargo:rustc-env=EMCC_CFLAGS=-sERROR_ON_UNDEFINED_SYMBOLS=0 --no-entry");
}