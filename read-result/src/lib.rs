//
// lib.rs
// Copyright (C) 2024 gum <gum@akoyono>
// Distributed under terms of the MIT license.
//

#![feature(hash_extract_if)]
use std::path::Path;
use anyhow::Result;
pub mod list_paths;
pub mod f2id_fs;

pub fn create_file_tensor<P: AsRef<Path>>(root_dir: P) -> Result<f2id_fs::Files> {
    let paths = list_paths::list_paths(&root_dir)?;
    Ok(f2id_fs::Files::new(paths))
}
