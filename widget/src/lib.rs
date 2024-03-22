// #![feature(hash_extract_if)]
use console_log;
use log::Level;
use wasm_bindgen::prelude::*;
use web_sys::HtmlElement;
use features2image_diffusion_read_result::f2id_fs::Files;
use serde_json;
use std::boxed::Box;

pub mod feature_images;
use feature_images::FeatureImages;

fn init_logger() {
    console_log::init_with_level(Level::Debug);
}


#[wasm_bindgen]
pub fn feature_images(root: &HtmlElement, pathor: String) {
    init_logger();
    root.set_inner_html("");
    dioxus_web::launch::launch(
        FeatureImages,
        vec![Box::new(move || {
            let files: Files = serde_json::from_str(&pathor).unwrap();
            Box::new(files)
        })],
        dioxus_web::Config::new().rootname(root.id())
    );
}
