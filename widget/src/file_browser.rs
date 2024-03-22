// This code is mostly copied from the dioxus repo.

use dioxus::prelude::*;
use serde_json;

#[component]
pub fn FileBrowser(cx: Scope) -> Element {
    use_shared_state_provider(cx, Files::new);
    use_shared_state_provider(cx, || SelectedFile { path: None });
    let selected_file = use_shared_state::<SelectedFile>(cx).unwrap();
    render!( div {
        class: "flex",
        div {
            class: "flex-1",
            FolderList {},
            FileList {},
        },
        {match &selected_file.read().path {
            Some(_) => rsx!( div {
                class: "w-1/2",
                FilePreview {},
            }),
            None => rsx!(div { class: "hidden" })
        }}
    })
}

struct SelectedFile {
    path: Option<String>,
}

#[component]
fn FilePreview(cx: Scope) -> Element {
    let selected_file = use_shared_state::<SelectedFile>(cx).unwrap();

    render!({
        match &selected_file.read().path {
            Some(path) => rsx!(embed {
                class: "max-w-full",
                src: "{path}"
            }),
            None => rsx!( h2 {
                class: "text-3xl",
                "weird, you shouldn't see this text"
            }),
        }
    })
}

#[component]
fn FileList(cx: Scope) -> Element {
    let files = use_shared_state::<Files>(cx).unwrap();
    let selected_file = use_shared_state::<SelectedFile>(cx).unwrap();

    let rendered_files = rsx!({
        files
            .read()
            .path_names
            .iter()
            .filter(|p| p.contains('.'))
            .map(|path: &String| {
                let p = path.clone();
                rsx!(button {
                    class: "m-1 p-1 bg-sky-500 rounded",
                    onclick: move |_| {
                        selected_file.write().path = Some(format!(
                            "{}/{}",
                            files.read().current_directory,
                            p
                        ));
                    },
                    "{path}"
                })
            })
    });

    render!(div {
        class: "inline-flex flex-wrap",
        { rendered_files }
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
                    class: "truncate bg-indigo-500 m-1 p-1 rounded-lg",
                    onclick: move |_| {
                        files.write().enter_dir(&path)
                    },
                    "{path}"
                })
            })
    });

    render!(div {
        class: "grid grid-cols-auto",
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
            all_paths: serde_json::from_str(include_str!("../pkg/files.json")).unwrap(),
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
            self.current_directory = self
                .current_directory
                .chars()
                .into_iter()
                .rev()
                .filter(|c| {
                    if !slash {
                        if *c == '/' {
                            slash = true;
                        }
                        false
                    } else {
                        true
                    }
                })
                .collect();
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
