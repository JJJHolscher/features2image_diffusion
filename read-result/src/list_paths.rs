use std::env::{set_current_dir, current_dir};
use std::fs;
use std::path::{Path, PathBuf};
use std::vec::IntoIter;

struct FileIterator {
    dirs: Vec<PathBuf>,
    current_entries: Option<IntoIter<PathBuf>>,
}

impl FileIterator {
    fn new<P: AsRef<Path>>(root: P) -> Self {
        FileIterator {
            dirs: vec![root.as_ref().to_path_buf()],
            current_entries: None,
        }
    }
}

impl Iterator for FileIterator {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut entries) = self.current_entries {
                // Try to yield the next file from the current directory
                if let Some(entry) = entries.next() {
                    return Some(entry);
                }
            }

            // Current directory is exhausted or not set, move to the next directory
            let next_dir = match self.dirs.pop() {
                Some(dir) => dir,
                None => return None, // No more directories to visit
            };

            let entries = match fs::read_dir(&next_dir) {
                Ok(entries) => entries,
                Err(_) => continue, // Skip directories we can't read
            };

            let files = entries
                .filter_map(|entry| {
                    entry.ok().map(|e| {
                        let path = e.path();
                        if path.is_dir() {
                            self.dirs.push(path.clone());
                        }
                        path
                    })
                })
                .collect::<Vec<_>>()
                .into_iter();

            self.current_entries = Some(files);
        }
    }
}

pub fn list_paths<P: AsRef<Path>>(root_dir: P) -> std::io::Result<Vec<PathBuf>> {
    let orig_dir = current_dir()?;
    set_current_dir(&root_dir)?;
    let paths = FileIterator::new("./").collect();
    set_current_dir(orig_dir)?;
    Ok(paths)
}
