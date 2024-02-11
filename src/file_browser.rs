// This code is mostly copied from the dioxus repo.

use core::ops::Deref;
use dioxus::prelude::*;
use jupyter_and_dioxus::DioxusInElement;
use serde_json;
use web_sys::HtmlElement;

#[cfg(not(feature = "collect-assets"))]
const _STYLE: &str = include_str!("../res/fileexplorer.css");

#[cfg(feature = "collect-assets")]
const _STYLE: &str = manganis::mg!(file("./res/fileexplorer.css"));

pub struct FileBrowser {
    inner_text: String,
}

impl DioxusInElement for FileBrowser {
    fn new(root: &HtmlElement) -> Self {
        let inner_text = root.inner_text();
        root.set_inner_html("");
        FileBrowser { inner_text }
    }

    fn component(cx: Scope<FileBrowser>) -> Element {
        let files = use_ref(cx, Files::new);


        cx.render(rsx!({
            files.read().path_names.iter().map(|full_path: &String| {
                let full_path: String = full_path.clone();
                let path: String = full_path.split('/').last().unwrap_or(full_path.as_str()).to_owned();
                if path.contains('.') {
                    rsx!(h1 { "{path}" })
                } else {
                    rsx!(button {
                        onclick: move |_| {files.write().enter_dir(&full_path)},
                        "{path}"
                    })
                }
            })
        }))
    }
}

struct Files {
    all_paths: Vec<String>,
    current_directory: String,
    path_names: Vec<String>,
    err: Option<String>,
}

impl Files {
    fn new() -> Self {
        let mut files = Self {
            all_paths: serde_json::from_str(include_str!("files.json")).unwrap(),
            current_directory: "./".to_string(),
            path_names: vec![],
            err: None,
        };

        files.reload_path_list();

        files
    }

    fn reload_path_list(&mut self) {
        self.path_names.clear();
        for path in &self.all_paths {
            if path.starts_with(&self.current_directory) {
                self.path_names.push(path.to_string());
            }
        }
    }

    fn go_up(&mut self) {
        if self.current_directory != "./" {
            let mut slash = false;
            // Trim all characters after the last slash.
            self.current_directory = self.current_directory.chars().into_iter().rev().filter(|c| {
                if !slash {
                    if *c == '/' {
                        slash = true;
                    }
                    false
                } else {
                    true
                }
            }).collect();
            self.reload_path_list();
        }
    }

    fn enter_dir(&mut self, path: &str) {
        self.current_directory = path.to_string();
        self.reload_path_list();
    }

    fn current(&self) -> &str {
        &self.current_directory
    }

    fn clear_err(&mut self) {
        self.err = None;
    }
}
