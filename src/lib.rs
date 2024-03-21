#![feature(hash_extract_if)]
use console_log;
use log::Level;
use wasm_bindgen::prelude::*;
use web_sys::HtmlElement;

mod f2id_fs;
mod file_browser;
use file_browser::FileBrowser;
mod feature_images;
use feature_images::{FeatureImages, FeatureImagesProps};

fn init_logger() {
    console_log::init_with_level(Level::Debug);
}

#[wasm_bindgen]
pub fn file_browser(root: &HtmlElement) {
    init_logger();
    dioxus_web::launch_cfg(
        FileBrowser,
        dioxus_web::Config::new().rootname(root.id())
    );
}


#[wasm_bindgen]
pub fn feature_images(root: &HtmlElement) {
    init_logger();
    let run_id = root.get_attribute("run").unwrap();
    root.set_inner_html("");
    dioxus_web::launch_with_props(
        FeatureImages,
        FeatureImagesProps { run_id },
        dioxus_web::Config::new().rootname(root.id())
    );
}
