use std::collections::HashSet;
use std::collections::HashMap;
use std::mem::discriminant;
use serde_json;
use itertools::Itertools;
use tao_log::{debugv, info};
use ndarray::prelude::*;
use ndarray::Slice;
use strum::{IntoEnumIterator, EnumIter};


#[derive(Hash, EnumIter, PartialEq, std::cmp::Eq, Ord, PartialOrd, Clone, Debug)]
pub enum Argument {
    Data(u16),
    Feature(u16),
    Modification(u16),
    FileType(String)
}

#[derive(Clone, PartialEq, Default, Debug)]
pub struct File {
    pub data: u16,
    pub feature: u16,
    pub modification: u16,
    pub file_type: String,
    pub path: String,
}

impl File {
    pub fn new(data: Argument, feature: Argument, modification: Argument, file_type: Argument, path: String) -> Self {
        let Argument::Data(data) = data else { todo!() };
        let Argument::Feature(feature) = feature else { todo!() };
        let Argument::Modification(modification) = modification else { todo!() };
        let Argument::FileType(file_type) = file_type else { todo!() };

        File {
            data,
            feature,
            modification,
            file_type,
            path
        }
    }
}

// #[derive(Debug, EnumIter, Hash, PartialEq, std::cmp::Eq)]
// pub enum Parameter {
    // Data,
    // Feature,
    // Modification,
    // FileType
// }

/// Responsible for storing all files assosiated with generating images that were conditioned on
/// particular features.
/// Every single image generation, was dependent on a couple of parameters:
/// data_id: the data point of which features were modified.
/// feature_id: the feature that was modified.
/// modification_id: the degree of modification of the feature.
/// file_type: type of the file generated during the image generation.
///
/// All the files are stored in a tensor that has as many dimensions as there are parameters.
/// Every element of the tensor is a path to a generated file.
/// For this to work, any value of an argument needs to map to an index in their dimension of the
/// tensor.
///
/// So, for retrieval purposes. There are 3 types of variables of note:
/// Parameter: A variable that determined what would be generated.
/// Argument: The value of the parameter.
/// Index: The mapping of an Parameter's Argument to a location in the tensor.
///
/// I should consider using fselect in the future
#[derive(PartialEq)]
pub struct Files {
    pub run: String,
    pub arg2idx: HashMap<Argument, usize>,
    pub idx2arg: HashMap<Argument, Vec<Argument>>,
    pub dfmt: Array4<File>,
}


impl Files {
    pub fn new(run_id: &str) -> Self {
        let paths: Vec<String> = serde_json::from_str(include_str!("../pkg/files.json")).unwrap();

        // Accept "./path" and "path".
        let run = if run_id.starts_with("./") {
            run_id.to_owned()
        } else {
            format!("./{run_id}")
        };

        // Collect all paths and their arguments.
        let mut all_arg: HashSet<Argument> = HashSet::new();
        let mut all = Vec::new();
        for path in paths.into_iter().filter(|p| p.starts_with(&run)) {
            let Some((d, f, m, t)) = arguments_from_path(&path) else { continue };
            
            all.push((path, d.clone(), f.clone(), m.clone(), t.clone()));

            all_arg.insert(d);
            all_arg.insert(f);
            all_arg.insert(m);
            all_arg.insert(t);

        }

        // Map arguments to indices and vice-versa.
        let mut idx2arg = HashMap::new();
        let mut arg2idx = HashMap::new();

        for parameter in Argument::iter() {
            let argument_set = all_arg.extract_if(|a| discriminant(a) == discriminant(&parameter));
            let mut arguments: Vec<Argument> = argument_set.into_iter().collect();
            arguments.sort();
            idx2arg.insert(parameter, arguments.clone());

            for (idx, arg) in arguments.into_iter().enumerate() {
                arg2idx.insert(arg, idx);
            }
        }

        // Fill the tensor with paths indexed by their arguments.
        let mut dfmt = Array4::default((
            idx2arg[&Argument::Data(0)].len(),
            idx2arg[&Argument::Feature(0)].len(),
            idx2arg[&Argument::Modification(0)].len(),
            idx2arg[&Argument::FileType(String::new())].len()
        ));
        for (path, d, f, m, t) in all {
            dfmt[[ arg2idx[&d], arg2idx[&f], arg2idx[&m], arg2idx[&t] ]] = File::new(
                d.clone(), f.clone(), m.clone(), t.clone(), path
            );
        }

        Self {
            run,
            idx2arg,
            arg2idx,
            dfmt
        }
    }

    pub fn get<'a>(&'a self, args: &[Argument]) -> ArrayView<'a, File, IxDyn>
    {
        let mut slice = self.dfmt.slice(s![.., .., .., ..]);
        for arg in args {
            match arg {
                Argument::Data(_) => {
                    let d = isize::try_from(self.arg2idx[&arg]).unwrap();
                    slice.slice_axis_inplace(Axis(0), Slice{start: d, end: Some(d+1), step: 1});
                },
                Argument::Feature(_) => {
                    let f = isize::try_from(self.arg2idx[&arg]).unwrap();
                    slice.slice_axis_inplace(Axis(1), Slice{start: f, end: Some(f+1), step: 1});
                },
                Argument::Modification(_) => {
                    let m = isize::try_from(self.arg2idx[&arg]).unwrap();
                    slice.slice_axis_inplace(Axis(2), Slice{start: m, end: Some(m+1), step: 1});
                },
                Argument::FileType(_) => {
                    let t = isize::try_from(self.arg2idx[&arg]).unwrap();
                    slice.slice_axis_inplace(Axis(3), Slice{start: t, end: Some(t+1), step: 1});
                },
            };
        }

        let mut out: ArrayView<'a, File, IxDyn> = slice.into_dyn();
        for (axis, size) in slice.shape().iter().enumerate().rev() {
            if out.shape().len() > 1 && *size == 1 {
                out = out.remove_axis(Axis(axis));
            }
        }
        out
    }
}


pub fn arguments_from_path(path: &str) -> Option<(Argument, Argument, Argument, Argument)> {
    let (".", _, d, fmt) = path.splitn(4, '/').next_tuple()? else { todo!() };
    let d = d.parse().ok()?;
    let (f, m, t) = match fmt.split_once('/') {
        // A valid path is either "./run/data/feature/modification-file_type"
        Some((f, mt)) => {
            let (m, t) = mt.split_once('-')?;
            Some((f.parse().ok()?, m.parse().ok()?, t.to_owned()))
        },
        // or "./run/data/'unedited'-file_type".
        None => {
            let ("unedited", t) = fmt.split_once('-')? else { todo!() };
            Some((0, 0, t.to_owned()))
        }
    }?;

    Some((
        Argument::Data(d),
        Argument::Feature(f),
        Argument::Modification(m),
        Argument::FileType(t)
    ))
}
