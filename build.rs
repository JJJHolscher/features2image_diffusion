use std::env::set_current_dir;
use std::fs::{self, ReadDir};
use std::io::Write;
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

            dbg!(&next_dir);

            let entries = match fs::read_dir(&next_dir) {
                Ok(entries) => entries,
                Err(_) => continue, // Skip directories we can't read
            };

            dbg!(&entries);

            let files = entries
                .filter_map(|entry| {
                    entry.ok().and_then(|e| {
                        let path = e.path();
                        if path.is_dir() {
                            self.dirs.push(path.clone());
                        }
                        Some(path)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter();

            dbg!(&files);

            self.current_entries = Some(files);
        }
    }
}

fn main() -> std::io::Result<()> {
    let project_dir = std::env::current_dir().unwrap();
    let root_path = "run/4409b6282a7d05f0b08880228d6d6564011fa40be412073ff05aff8bf2dc49fa";
    set_current_dir(Path::new(root_path))?;
    dbg!(fs::read_dir(std::env::current_dir().unwrap()));
    let file_paths: Vec<PathBuf> = FileIterator::new("./").collect();
    dbg!(&file_paths);

    // Serialize and write to file as before
    let serialized = serde_json::to_string(
        &file_paths
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<String>>(),
    )
    .unwrap();
    set_current_dir(project_dir)?;
    std::fs::File::create("src/files.json")?.write_all(serialized.as_bytes())?;

    Ok(())
}
