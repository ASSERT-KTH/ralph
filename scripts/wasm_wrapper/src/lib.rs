use wasm_bindgen::prelude::*;
use ort::{GraphOptimizationLevel, Session};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Embed the model here
static MODEL: &[u8] = include_bytes!("model.onnx");

// Load the model in the start function
static mut model: Option<ort::Session> = None;

#[wasm_bindgen(start)]
pub fn load_model(){
    log("Loading model...");
    unsafe  {
        model = Some(
        ort::Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
            .with_intra_threads(4).unwrap()
            .with_model_from_memory(&MODEL).unwrap()
        );
    }
    log("Model loaded!");
}

#[wasm_bindgen]
pub fn infer(wasm_bytes: &[u8]){
    // Turn the bytes into a vector of 100x100 integers
    
}

pub fn main() {
    println!("Hello, world!");
}