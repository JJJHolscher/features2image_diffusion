use dioxus::prelude::*;
use jupyter_and_dioxus::DioxusInElement;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlElement;

mod file_browser;

#[wasm_bindgen]
pub fn file_browser(elem: &HtmlElement) {
    file_browser::FileBrowser::launch(elem);
}
