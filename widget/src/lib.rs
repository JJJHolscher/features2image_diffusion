// #![feature(hash_extract_if)]
use console_log;
use log::Level;
use wasm_bindgen::prelude::*;
use web_sys::HtmlElement;
use features2image_diffusion_read_result::f2id_fs::Files;
use serde_json;
use std::boxed::Box;
use tao_log::{debugv, info};

pub mod feature_images;
use feature_images::FeatureImages;

fn init_logger() {
    console_log::init_with_level(Level::Debug);
    console_error_panic_hook::set_once();
}


#[wasm_bindgen]
pub fn feature_images(root: &HtmlElement, run_dir: String, files: Vec<u8>) {
    init_logger();
    root.set_inner_html("");
    debugv!(&files);
    dioxus_web::launch::launch(
        FeatureImages,
        vec![
            Box::new(move || Box::new(
                Files::from_bytes(&files)
                    .expect("could not deserialize the files")
            )),
            Box::new(move || Box::new(
                run_dir.clone()
            )),
        ],
        dioxus_web::Config::new().rootname(root.id())
    );
}
