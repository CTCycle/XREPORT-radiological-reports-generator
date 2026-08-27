#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod navigation;
mod runtime;
mod single_instance;
mod windows_job;

use backend::BackendHandle;
use single_instance::SingleInstance;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{Manager, Url, WebviewWindow, WindowEvent};

const RELEASE_VERSION: &str = env!("CARGO_PKG_VERSION");
const RUNTIME_VARIANT: &str = env!("XREPORT_DESKTOP_VARIANT");

#[derive(Clone)]
struct DesktopState {
    backend: Arc<Mutex<Option<Arc<BackendHandle>>>>,
    data_root: PathBuf,
}

impl DesktopState {
    fn new(data_root: PathBuf) -> Self {
        Self {
            backend: Arc::new(Mutex::new(None)),
            data_root,
        }
    }

    fn take_backend(&self) -> Option<Arc<BackendHandle>> {
        self.backend.lock().ok().and_then(|mut value| value.take())
    }
}

fn data_root() -> PathBuf {
    env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
        .join("XREPORT")
        .join("data")
}

fn log_shell(data_root: &Path, message: &str) {
    let directory = data_root.join("logs");
    let _ = std::fs::create_dir_all(&directory);
    let path = directory.join("desktop-shell.log");
    if path
        .metadata()
        .map(|metadata| metadata.len() > 2 * 1024 * 1024)
        .unwrap_or(false)
    {
        let first = directory.join("desktop-shell.log.1");
        let second = directory.join("desktop-shell.log.2");
        let _ = std::fs::remove_file(&second);
        let _ = std::fs::rename(&first, &second);
        let _ = std::fs::rename(&path, &first);
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        use std::io::Write;
        let _ = writeln!(file, "{}", message.replace('\n', " "));
    }
}

#[cfg(target_os = "windows")]
const STARTUP_ERROR_BASE_URL: &str = "http://tauri.localhost";
#[cfg(not(target_os = "windows"))]
const STARTUP_ERROR_BASE_URL: &str = "tauri://localhost";

fn startup_error_url(data_root: &Path, message: &str) -> Url {
    let mut url =
        Url::parse(&format!("{STARTUP_ERROR_BASE_URL}/error.html")).expect("static error URL");
    url.query_pairs_mut()
        .append_pair("message", message)
        .append_pair(
            "log",
            &data_root
                .join("logs")
                .join("desktop-shell.log")
                .to_string_lossy(),
        );
    url
}

fn show_startup_error(window: &WebviewWindow, data_root: &Path, message: &str) {
    log_shell(data_root, &format!("startup failure: {message}"));
    if let Err(error) = window.navigate(startup_error_url(data_root, message)) {
        log_shell(
            data_root,
            &format!("startup error page navigation failed: {error}"),
        );
    }
}

fn navigate_to_backend(window: &WebviewWindow, handle: &BackendHandle) -> Result<(), String> {
    let url = Url::parse(&handle.bootstrap_url()).map_err(|error| error.to_string())?;
    if !navigation::is_allowed_navigation(&url, Some(handle.port)) {
        return Err("refusing to navigate to an untrusted backend URL".to_string());
    }
    window.navigate(url).map_err(|error| error.to_string())
}

fn start_packaged_backend(window: WebviewWindow, state: DesktopState) {
    let backend_store = state.backend.clone();
    let data_root = state.data_root.clone();
    thread::spawn(move || {
        let _ = window.eval("document.getElementById('status').textContent = 'Extracting the verified local runtime…';");
        match backend::start_backend(&data_root, RELEASE_VERSION, RUNTIME_VARIANT) {
            Ok(handle) => {
                let handle = Arc::new(handle);
                if let Ok(mut slot) = backend_store.lock() {
                    *slot = Some(handle.clone());
                }
                if let Err(error) = navigate_to_backend(&window, &handle) {
                    show_startup_error(&window, &data_root, &error);
                    if let Ok(mut slot) = backend_store.lock() {
                        slot.take();
                    }
                    return;
                }

                let monitor_window = window.clone();
                let monitor_store = backend_store.clone();
                let monitor_data_root = data_root.clone();
                thread::spawn(move || loop {
                    thread::sleep(Duration::from_millis(750));
                    let Some(handle) = monitor_store.lock().ok().and_then(|slot| slot.clone())
                    else {
                        break;
                    };
                    match handle.has_exited() {
                        Ok(Some(code)) => {
                            show_startup_error(
                                &monitor_window,
                                &monitor_data_root,
                                &format!(
                                    "The packaged backend stopped unexpectedly (exit code {code})."
                                ),
                            );
                            if let Ok(mut slot) = monitor_store.lock() {
                                slot.take();
                            }
                            break;
                        }
                        Ok(None) => {}
                        Err(error) => {
                            log_shell(
                                &monitor_data_root,
                                &format!("backend watchdog error: {error}"),
                            );
                            break;
                        }
                    }
                });
            }
            Err(error) => show_startup_error(&window, &data_root, &error),
        }
    });
}

pub fn run() {
    let state = DesktopState::new(data_root());
    let instance = match SingleInstance::acquire() {
        Ok(Some(instance)) => instance,
        Ok(None) => return,
        Err(error) => {
            eprintln!("Unable to acquire XREPORT desktop instance guard: {error}");
            return;
        }
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(state.clone())
        .manage(instance)
        .setup(move |app| {
            let window = app
                .get_webview_window("main")
                .ok_or_else(|| "configured XREPORT main window is missing".to_string())?;
            if cfg!(debug_assertions) && env::var("XREPORT_DESKTOP_DEV").is_ok() {
                let _ = window.show();
                return Ok(());
            }
            start_packaged_backend(window, state.clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::CloseRequested { .. }) {
                let state = window.state::<DesktopState>().inner().clone();
                if let Some(backend) = state.take_backend() {
                    backend.stop();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running XREPORT desktop application");
}

#[cfg(test)]
mod tests {
    use super::startup_error_url;
    use std::path::Path;

    #[test]
    fn startup_error_url_uses_the_embedded_asset_origin_and_encodes_details() {
        let url = startup_error_url(Path::new(r"C:\XREPORT\data"), "runtime failed & stopped");

        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(url.scheme(), "tauri");
            assert_eq!(url.host_str(), Some("localhost"));
        }
        #[cfg(target_os = "windows")]
        {
            assert_eq!(url.scheme(), "http");
            assert_eq!(url.host_str(), Some("tauri.localhost"));
        }
        assert_eq!(url.path(), "/error.html");
        assert_eq!(
            url.query_pairs()
                .find(|(key, _)| key == "message")
                .unwrap()
                .1,
            "runtime failed & stopped"
        );
        let expected_log = Path::new(r"C:\XREPORT\data")
            .join("logs")
            .join("desktop-shell.log");
        assert_eq!(
            url.query_pairs().find(|(key, _)| key == "log").unwrap().1,
            expected_log.to_string_lossy()
        );
    }
}
