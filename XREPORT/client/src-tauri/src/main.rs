#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tauri::{Manager, RunEvent};

struct BackendChildState {
    child: Arc<Mutex<Option<Child>>>,
}

fn parse_dotenv_value(raw: &str) -> String {
    let trimmed = raw.trim();
    let unquoted = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };
    unquoted.to_string()
}

fn resolve_backend_host_port(env_path: &Path) -> (String, u16) {
    let mut host = String::from("127.0.0.1");
    let mut port = 8000u16;

    let content = match fs::read_to_string(env_path) {
        Ok(value) => value,
        Err(_) => return (host, port),
    };

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let key = key.trim();
        let value = parse_dotenv_value(value);
        if key.eq_ignore_ascii_case("FASTAPI_HOST") && !value.is_empty() {
            host = value;
        } else if key.eq_ignore_ascii_case("FASTAPI_PORT") {
            if let Ok(parsed) = value.parse::<u16>() {
                port = parsed;
            }
        }
    }

    if host == "0.0.0.0" {
        host = String::from("127.0.0.1");
    }

    (host, port)
}

fn can_connect(host: &str, port: u16) -> bool {
    let address = format!("{host}:{port}");
    let Ok(addrs) = address.to_socket_addrs() else {
        return false;
    };

    for addr in addrs {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok() {
            return true;
        }
    }

    false
}

fn js_escape(raw: &str) -> String {
    raw.replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "")
}

fn render_startup_error(app_handle: &tauri::AppHandle, message: &str) {
    let escaped = js_escape(message);
    let script = format!(
        "document.body.style.background='#0f172a';document.body.style.color='#e2e8f0';document.body.style.fontFamily='Segoe UI, sans-serif';document.body.style.padding='20px';document.body.innerHTML='<h2>XREPORT Desktop startup error</h2><pre style=\"white-space:pre-wrap;line-height:1.4;\">{escaped}</pre>';"
    );

    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.eval(&script);
    }
}

fn find_launcher_path(app_handle: &tauri::AppHandle) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(resource_dir) = app_handle.path().resource_dir() {
        candidates.push(resource_dir.join("XREPORT").join("start_on_windows_tauri.bat"));
        candidates.push(resource_dir.join("start_on_windows_tauri.bat"));
    }

    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir.join("start_on_windows_tauri.bat"));
        candidates.push(current_dir.join("..").join("start_on_windows_tauri.bat"));
        candidates.push(current_dir.join("..").join("..").join("start_on_windows_tauri.bat"));
        candidates.push(current_dir.join("XREPORT").join("start_on_windows_tauri.bat"));
    }

    candidates.into_iter().find(|candidate| candidate.is_file())
}

fn spawn_backend(app_handle: &tauri::AppHandle, state: &BackendChildState) -> Result<(), String> {
    #[cfg(not(target_os = "windows"))]
    {
        let _ = app_handle;
        let _ = state;
        return Err(String::from("Local Mode (v2) is currently supported on Windows only."));
    }

    #[cfg(target_os = "windows")]
    {
        let launcher = find_launcher_path(app_handle)
            .ok_or_else(|| String::from("Cannot find start_on_windows_tauri.bat in app resources or workspace."))?;

        let project_dir = launcher
            .parent()
            .ok_or_else(|| String::from("Launcher path has no parent directory."))?
            .to_path_buf();

        let env_path = project_dir.join("settings").join(".env");
        let (backend_host, backend_port) = resolve_backend_host_port(&env_path);

        let launcher_str = launcher.to_string_lossy().to_string();
        let child = Command::new("cmd")
            .arg("/c")
            .arg(launcher_str)
            .arg("--backend")
            .current_dir(project_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("Failed to start backend launcher: {error}"))?;

        {
            let mut guard = state
                .child
                .lock()
                .map_err(|_| String::from("Failed to lock backend child state."))?;
            *guard = Some(child);
        }

        let app_handle_clone = app_handle.clone();
        thread::spawn(move || {
            let max_attempts = 120u32;
            for _ in 0..max_attempts {
                if can_connect(&backend_host, backend_port) {
                    let url = format!("http://{backend_host}:{backend_port}/");
                    let script = format!("window.location.replace('{}');", js_escape(&url));
                    if let Some(window) = app_handle_clone.get_webview_window("main") {
                        let _ = window.eval(&script);
                    }
                    return;
                }
                thread::sleep(Duration::from_secs(1));
            }

            let timeout_message = format!(
                "Timed out waiting for backend at http://{backend_host}:{backend_port}/.\\nCheck XREPORT/settings/.env and backend bootstrap prerequisites."
            );
            render_startup_error(&app_handle_clone, &timeout_message);
        });

        Ok(())
    }
}

fn stop_backend(state: &BackendChildState) {
    if let Ok(mut guard) = state.child.lock() {
        if let Some(child) = guard.as_mut() {
            #[cfg(target_os = "windows")]
            {
                // Ensure cmd + uv/python descendants are terminated when the app exits.
                let _ = Command::new("taskkill")
                    .args(["/PID", &child.id().to_string(), "/T", "/F"])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
            }

            #[cfg(not(target_os = "windows"))]
            {
                let _ = child.kill();
            }

            let _ = child.wait();
        }
        *guard = None;
    }
}

fn main() {
    let backend_child = Arc::new(Mutex::new(None));
    let managed_state = BackendChildState {
        child: backend_child.clone(),
    };

    let app = tauri::Builder::default()
        .manage(managed_state)
        .setup(|app| {
            let Some(state) = app.try_state::<BackendChildState>() else {
                render_startup_error(app.handle(), "Internal startup error: missing backend state.");
                return Ok(());
            };

            if let Err(error) = spawn_backend(app.handle(), state.inner()) {
                render_startup_error(app.handle(), &error);
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |_app_handle, event| match event {
        RunEvent::Exit => {
            let state = BackendChildState {
                child: backend_child.clone(),
            };
            stop_backend(&state);
        }
        RunEvent::ExitRequested { .. } => {
            let state = BackendChildState {
                child: backend_child.clone(),
            };
            stop_backend(&state);
        }
        _ => {}
    });
}
