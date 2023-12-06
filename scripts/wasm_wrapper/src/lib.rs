use ort::{inputs, GraphOptimizationLevel, Session, SessionOutputs};
use ndarray::{s, Array, Axis};

// Embed the model here
static MODEL: &[u8] = include_bytes!("model.onnx");

// Load the model in the start function
static mut model: Option<ort::Session> = None;

// FIXME: this should be optimized with Wizer, in order to avoid the loading of the data on every fresh
// spawn of a potential wasm32-wasi module :)
#[no_mangle]
pub extern "C" fn init_model(){
    unsafe  {
        // puts(b"Init model...!\0".as_ptr());
        model = Some(
        ort::Session::builder().unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3).unwrap()
            // Sadly we cannot add several threads here due to that this will be loaded in a single Wasm thread
            .with_model_from_memory(&MODEL).unwrap()
        );
        // puts(b"Model ready...!\0".as_ptr());
    }
}


#[no_mangle]
pub extern "C" fn print_metadata() {
    unsafe {
       
        let session = model.as_ref().unwrap();
        let meta = session.metadata().unwrap();

        if let Ok(x) = meta.name() {
            println!("Name: {x}");
        }
        if let Ok(x) = meta.description() {
            println!("Description: {x}");
        }
        if let Ok(x) = meta.producer() {
            println!("Produced by {x}");
        }

        println!("Inputs:");
        for (i, input) in session.inputs.iter().enumerate() {
            println!("    {i} {}: {:?}", input.name, &input.input_type);
        }
        println!("Outputs:");
        for (i, output) in session.outputs.iter().enumerate() {
            println!("    {i} {}: {:?}", output.name, &output.output_type);
        }
    }
}

#[no_mangle]
pub extern "C" fn infer(wasm_ptr: *const u8, size: usize) -> i32  {
    // Turn the bytes into a vector of 100x100 integers
    let wasm_bytes = unsafe { std::slice::from_raw_parts(wasm_ptr, size) };
    println!("Inferring for len {}", wasm_bytes.len());
    let sqrt = (wasm_bytes.len() as f64).sqrt() as usize;
    if sqrt == 0 {
        println!("Invalid input len 0");
        return 1;
    }

    // Create an image from the bytes using sqrt*sqrt size
    // It is grayscale, so we can use a single channel
    println!("Creating image from the binary {}", sqrt);
    let img = image::GrayImage::from_raw(sqrt as u32, sqrt as u32, wasm_bytes.to_vec()).unwrap();
    // Now scale it to 100x100
    let img = image::imageops::resize(&img, 100, 100, image::imageops::FilterType::Nearest);
    // Convert it to a vector of floats
    let img = img.into_raw().iter().map(|x| *x as f32).collect::<Vec<f32>>();

    // Call the inferring
    if unsafe { model.is_none() } {
        println!("Model not initialized");
        return 1;
    }

    println!("Feeding input");
    let mut input = Array::zeros((1, 100*100));
    input.assign(&Array::from_shape_vec((1, 100*100), img).unwrap());

    let output = unsafe { model.as_ref().unwrap().run(inputs!["reshape_input" => input.view()].unwrap()).unwrap() };
	let output = output["dense"].extract_tensor::<f32>().unwrap().view().t().into_owned();
    let output = output.iter().map(|x| *x as f64).collect::<Vec<f64>>();

    println!("Output: {:?}", output);
    if output[0] > 0.5 {
        return 0;
    } else {
        return 1;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta() {
        init_model();
        print_metadata();

    }

    #[test]
    fn test_infer() {
        init_model();
        let bytes = include_bytes!("../tests/benign.wasm");
        infer(bytes.as_ptr(), bytes.len());
    }

    #[test]
    fn test_infer2() {
        init_model();
        let bytes = include_bytes!("../tests/malign.wasm");
        infer(bytes.as_ptr(), bytes.len());
    }
}
