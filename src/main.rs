mod file_browser;
use file_browser::FileBrowser;

fn main() {
    // launch the web app
    dioxus_web::launch_with_props(
        FileBrowser::component,
        FileBrowser{ inner_text: "".to_owned() },
        dioxus_web::Config::new().rootname("file_browser")
    );
}
