use std::ops::Deref;
use std::collections::HashSet;
use dioxus::prelude::*;

use tao_log::{debugv, info};
use crate::f2id_fs::{File, Files, Argument};


#[derive(PartialEq)]
enum SelectedParameter {
    Data,
    Feature,
    Modification
}


#[derive(Default)]
struct Arguments {
    pub data: u16,
    pub feature: u16,
    pub modification: u16
}


impl Arguments {
    fn to_index(&self, exclude: Option<&SelectedParameter>) -> Vec<Argument> {
        match exclude {
            Some(&SelectedParameter::Data) => {
                vec![Argument::Feature(self.feature), Argument::Modification(self.modification)]
            },
            Some(&SelectedParameter::Feature) => {
                vec![Argument::Data(self.data), Argument::Modification(self.modification)]
            },
            Some(&SelectedParameter::Modification) => {
                vec![Argument::Data(self.data), Argument::Feature(self.feature)]
            }
            None => {
                vec![Argument::Data(self.data), Argument::Feature(self.feature), Argument::Modification(self.modification)]
            }
        }
    }

    fn set(&mut self, parameter: &SelectedParameter, file: &File) {
        match parameter {
            &SelectedParameter::Data => {
                self.data = file.data.clone();
            },
            &SelectedParameter::Feature => {
                self.feature = file.feature.clone();
            },
            &SelectedParameter::Modification => {
                self.modification = file.modification.clone();
            }
        }
    }
}

#[component]
pub fn FeatureImages(cx: Scope, run_id: String) -> Element {
    let files = use_state(cx, || Files::new(run_id));

    // Track the selected arguments.
    use_shared_state_provider(cx, Arguments::default);
    use_shared_state_provider(cx, || SelectedParameter::Data);

    let mut args = use_shared_state::<Arguments>(cx).unwrap();
    let mut selection = use_shared_state::<SelectedParameter>(cx).unwrap();

    let [num_data, num_features, num_modifs, _] = files.dfmt.shape() else { todo!{} };

    let mut index = args.read().deref().to_index(None);
    index.push(Argument::FileType("images.png".to_owned()));
    let img_path = &files.get().get(&index)[0].path;

    render!(div {
        class: "flex flex-col justify-center",
        img {
            class: "",
            src: "/run/{img_path}"
        }
        SelectParameter{}
        SelectArgument{files: files.get()}
    })
}

#[component]
fn SelectParameter(cx: Scope) -> Element {
    let mut args = use_shared_state::<Arguments>(cx).unwrap();
    let mut selection = use_shared_state::<SelectedParameter>(cx).unwrap();
    render!(div {
        class: "",

        button {
            class: "truncate bg-indigo-500 m-1 p-1 rounded-lg",
            onclick: move |_| {
                *selection.write() = SelectedParameter::Data;
            },
            "data = {args.read().data}"
        }

        button {
            class: "truncate bg-indigo-500 m-1 p-1 rounded-lg",
            onclick: move |_| {
                *selection.write() = SelectedParameter::Feature;
            },
            "feature = {args.read().feature}"
        }

        button {
            class: "truncate bg-indigo-500 m-1 p-1 rounded-lg",
            onclick: move |_| {
                *selection.write() = SelectedParameter::Modification;
            },
            "modification = {args.read().modification}"
        }
    })
}

#[component]
fn SelectArgument<'a>(cx: Scope, files: &'a Files) -> Element {
    let selection = use_shared_state::<SelectedParameter>(cx).unwrap();
    let mut args = use_shared_state::<Arguments>(cx).unwrap();
    let options = files.get(&args.read().to_index(Some(selection.read().deref())));
    let mut added_options = HashSet::<u16>::new();
    render!(div {
        class: "grid grid-cols-10",
        {
            options
                .iter()
                .filter_map(|file: &File| {
                    let file = file.clone();
                    let option = match selection.read().deref() {
                        &SelectedParameter::Data => file.data,
                        &SelectedParameter::Feature => file.feature,
                        &SelectedParameter::Modification => file.modification,
                    };
                    if added_options.contains(&option) { None } else {
                        added_options.insert(option.clone());
                        Some(rsx!( button {
                            class: "truncate bg-indigo-500 m-1 p-1 rounded-lg",
                            onclick: move |_| {
                                args.write().set(selection.read().deref(), &file);
                            },
                            "{option}"
                        }))
                    }
                })
        }
    })
}

// #[component]
// fn IdOptions(cx: Scope, parameter: String) -> Element {

// }
