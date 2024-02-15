// This code is mostly copied from the dioxus repo.

use core::ops::Deref;
use dioxus::prelude::*;
// use jupyter_and_dioxus::DioxusInElement;
use serde_json;
use web_sys::HtmlElement;

pub struct FileBrowser {
    pub inner_text: String,
}

impl FileBrowser {
    fn new(root: &HtmlElement) -> Self {
        let inner_text = root.inner_text();
        root.set_inner_html("");
        FileBrowser { inner_text }
    }

    pub fn component(cx: Scope<FileBrowser>) -> Element {
        use_shared_state_provider(cx, Files::new);
        use_shared_state_provider(cx, || SelectedFile { path : None } );
        render!( div {
            class: "flex",
            div {
                class: "w-1/2",
                FolderList {},
                FileList {},
            },
            div {
                class: "w-1/2",
                FilePreview {},
            },
        })
    }

    pub fn launch(root: &HtmlElement) {
        dioxus_web::launch_with_props(
            Self::component,
            Self::new(root),
            dioxus_web::Config::new().rootname(root.id())
        );
    }
}


struct SelectedFile {
    path: Option<String>
}


#[component]
fn FilePreview(cx: Scope) -> Element {
    let files = use_shared_state::<Files>(cx).unwrap();
    let selected_file = use_shared_state::<SelectedFile>(cx).unwrap();

    match &selected_file.read().path {
        Some(path) => render!( embed {
            class: "text-3xl",
            src: "{files.read().current_directory}/{path}"
        }),
        None => render!( h2 {
            class: "text-3xl",
            "no selection"
        })
    }
}


#[component]
fn FileList(cx: Scope) -> Element {
    let files = use_shared_state::<Files>(cx).unwrap();
    let selected_file = use_shared_state::<SelectedFile>(cx).unwrap();

    render!({
        files.read().path_names.iter().filter(|p| p.contains('.')).map(|path: &String| {
            let p = path.clone();
            rsx!(button { 
                class: "",
                onclick: move |_| {
                    selected_file.write().path = Some(p.to_owned());
                },
                "{path}" 
            })
        })
    })
}


#[component]
fn FolderList(cx: Scope) -> Element {
    let files = use_shared_state::<Files>(cx).unwrap();

    let button_list = rsx!({
        files
            .read()
            .path_names
            .iter()
            .filter(|p| !p.contains('.'))
            .map(|path: &String| {
                let path = path.clone();
                rsx!( button {
                    class: "",
                    onclick: move |_| {
                        files.write().enter_dir(&path)
                    },
                    "{path}"
                })
            })
    });

    render!(div {
        class: "grid grid-cols-10 auto-rows-min ",
        { button_list }
    })
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
        let len = self.current_directory.len();
        for path in &self.all_paths {
            if path.starts_with(&self.current_directory) {
                let tail = &path[len..];
                if !tail.contains("/") {
                    self.path_names.push(tail.to_string());
                }
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
        self.current_directory = format!("{}{}/", self.current_directory, path);
        self.reload_path_list();
    }

    fn current(&self) -> &str {
        &self.current_directory
    }

    fn clear_err(&mut self) {
        self.err = None;
    }
}
