// This code is mostly copied from the dioxus repo.

use core::ops::Deref;
use dioxus::prelude::*;
use dioxus_signals::use_signal;
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
        let mut files = use_signal(&cx, Files::new);
        let files_rendered = files.read().path_names.iter().enumerate().map( |(dir_id, path)| {
            let path_end = path.split('/').last().unwrap_or(path.as_str());
            rsx! ( div {
                class: "folder",
                key: "{path}",
                i { class: "material-icons",
                    onclick: move |_| files.write().enter_dir(dir_id),
                    if path_end.contains('.') {
                        "description"
                    } else {
                        "folder"
                    }
                    p { class: "cooltip", "0 folders / 0 files" }
                }
                h1 { "{path_end}" }
            })
        });
        cx.render(rsx! ( 
                {files_rendered}
        ))
            // link { href:"https://fonts.googleapis.com/icon?family=Material+Icons", rel:"stylesheet" }
            // header {
                // i { class: "material-icons icon-menu", "menu" }
                // h1 { "Files: ", {files.read().current()} }
                // span { }
                // i { class: "material-icons", onclick: move |_| files.write().go_up(), "logout" }
            // }
            // style { "{_STYLE}" }
            // main {
            // }
    }
}

struct Files {
    all_paths: Vec<String>,
    path_stack: Vec<String>,
    path_names: Vec<String>,
    err: Option<String>,
}

impl Files {
    fn new() -> Self {
        let mut files = Self {
            all_paths: serde_json::from_str(include_str!("files.json")).unwrap(),
            path_stack: vec!["./".to_string()],
            path_names: vec![],
            err: None,
        };

        files.reload_path_list();

        files
    }

    fn reload_path_list(&mut self) {
        let cur_path = self.path_stack.last().unwrap();
        self.path_names.clear();
        for path in &self.all_paths {
            if path.starts_with(cur_path) {
                self.path_names.push(path.to_string());
            }
        }
    }

    fn go_up(&mut self) {
        if self.path_stack.len() > 1 {
            self.path_stack.pop();
        }
        self.reload_path_list();
    }

    fn enter_dir(&mut self, dir_id: usize) {
        let path = &self.path_names[dir_id];
        self.path_stack.push(path.clone());
        self.reload_path_list();
    }

    fn current(&self) -> &str {
        self.path_stack.last().unwrap()
    }
    fn clear_err(&mut self) {
        self.err = None;
    }
}
