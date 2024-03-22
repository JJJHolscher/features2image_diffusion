// #![feature(hash_extract_if)]
use console_log;
use log::Level;
use wasm_bindgen::prelude::*;
use web_sys::HtmlElement;
use features2image_diffusion_read_result::f2id_fs::Files;
use serde_json;

pub mod feature_images;
use feature_images::{FeatureImages, FeatureImagesProps};

fn init_logger() {
    console_log::init_with_level(Level::Debug);
}


#[wasm_bindgen]
pub fn feature_images(root: &HtmlElement, pathor: String) {
    init_logger();
    let pathor: Files = serde_json::from_str(&pathor).unwrap();
    root.set_inner_html("");
    dioxus_web::launch::launch(
        FeatureImages,
        vec![||pathor],
        dioxus_web::Config::new().rootname(root.id())
    );
}
