#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashSet;
use std::fs;
use std::net::{TcpStream, ToSocketAddrs};
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent};

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[derive(Clone)]
struct BackendChildState {
    child: Arc<Mutex<Option<Child>>>,
}

struct BackendLaunchConfig {
    host: String,
    port: u16,
    reload: bool,
    install_extras: bool,
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

fn parse_boolish(raw: &str) -> bool {
    matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn resolve_backend_launch_config(env_path: &Path) -> BackendLaunchConfig {
    let mut config = BackendLaunchConfig {
        host: String::from("127.0.0.1"),
        port: 8000u16,
        reload: false,
        install_extras: false,
    };

    let content = match fs::read_to_string(env_path) {
        Ok(value) => value,
        Err(_) => return config,
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
            config.host = value;
        } else if key.eq_ignore_ascii_case("FASTAPI_PORT") {
            if let Ok(parsed) = value.parse::<u16>() {
                config.port = parsed;
            }
        } else if key.eq_ignore_ascii_case("RELOAD") {
            config.reload = parse_boolish(&value);
        } else if key.eq_ignore_ascii_case("OPTIONAL_DEPENDENCIES") {
            config.install_extras = parse_boolish(&value);
        }
    }

    if config.host == "0.0.0.0" {
        config.host = String::from("127.0.0.1");
    }

    config
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

fn render_startup_screen(
    app_handle: &tauri::AppHandle,
    title: &str,
    message: &str,
    accent: &str,
    show_spinner: bool,
) {
    let escaped_title = js_escape(title);
    let escaped_message = js_escape(message);
    let spinner_markup = if show_spinner {
        format!(
            "<div style='display:flex;align-items:center;gap:14px;margin:0 0 20px 0;'><div style='width:22px;height:22px;border:3px solid rgba(226,232,240,0.20);border-top-color:{accent};border-right-color:{accent};border-radius:50%;animation:xreport-spin 0.85s linear infinite;'></div><div style='font-size:12px;letter-spacing:0.16em;text-transform:uppercase;color:#94a3b8;'>Launching services</div></div>"
        )
    } else {
        String::new()
    };

    let script = format!(
        "document.body.style.margin='0';document.body.style.background='radial-gradient(circle at top, #16213a 0%, #0f172a 58%, #020617 100%)';document.body.style.color='#e2e8f0';document.body.style.fontFamily='Segoe UI, sans-serif';document.body.innerHTML=\"<style>@keyframes xreport-spin{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}</style><div style='min-height:100vh;display:flex;align-items:center;justify-content:center;padding:32px;box-sizing:border-box;'><div style='max-width:720px;width:100%;background:rgba(15,23,42,0.82);backdrop-filter:blur(12px);border:1px solid rgba(148,163,184,0.20);border-radius:22px;padding:32px 34px;box-shadow:0 28px 70px rgba(0,0,0,0.42);'><div style='width:64px;height:6px;border-radius:999px;background:{accent};margin-bottom:20px;'></div>{spinner_markup}<h2 style='margin:0 0 12px 0;font-size:30px;font-weight:600;'>\"+ '{escaped_title}' +\"</h2><pre style='margin:0;white-space:pre-wrap;line-height:1.6;font-size:15px;color:#cbd5e1;'>\"+ '{escaped_message}' +\"</pre></div></div>\";"
    );

    if let Some(window) = app_handle.get_webview_window("main") {
        let _ = window.eval(&script);
    }
}

fn render_startup_status(app_handle: &tauri::AppHandle, message: &str) {
    render_startup_screen(
        app_handle,
        "Starting XREPORT Desktop",
        message,
        "#38bdf8",
        true,
    );
}

fn render_startup_error(app_handle: &tauri::AppHandle, message: &str) {
    render_startup_screen(
        app_handle,
        "XREPORT Desktop startup error",
        message,
        "#f87171",
        false,
    );
}

fn is_workspace_root(candidate: &Path) -> bool {
    candidate.join("pyproject.toml").is_file()
        && candidate
            .join("XREPORT")
            .join("server")
            .join("app.py")
            .is_file()
}

fn has_workspace_venv(candidate: &Path) -> bool {
    candidate
        .join(".venv")
        .join("Scripts")
        .join("python.exe")
        .is_file()
}

fn push_with_ancestors(base: &Path, candidates: &mut Vec<PathBuf>) {
    let mut cursor = Some(base);
    while let Some(path) = cursor {
        candidates.push(path.to_path_buf());
        cursor = path.parent();
    }
}

fn find_workspace_root(app_handle: &tauri::AppHandle) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(resource_dir) = app_handle.path().resource_dir() {
        push_with_ancestors(&resource_dir, &mut candidates);
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            push_with_ancestors(exe_dir, &mut candidates);
            push_with_ancestors(&exe_dir.join("resources"), &mut candidates);
        }
    }

    if let Ok(current_dir) = std::env::current_dir() {
        push_with_ancestors(&current_dir, &mut candidates);
    }

    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut workspace_candidates: Vec<PathBuf> = Vec::new();
    for candidate in candidates {
        if seen.insert(candidate.clone()) && is_workspace_root(&candidate) {
            workspace_candidates.push(candidate);
        }
    }

    if let Some(with_venv) = workspace_candidates
        .iter()
        .find(|candidate| has_workspace_venv(candidate))
    {
        return Some(with_venv.clone());
    }

    workspace_candidates.into_iter().next()
}

fn directory_is_writable(path: &Path) -> bool {
    if fs::create_dir_all(path).is_err() {
        return false;
    }

    let probe_path = path.join(".xreport-write-probe");
    let wrote_probe = fs::write(&probe_path, b"ok").is_ok();
    let _ = fs::remove_file(&probe_path);
    wrote_probe
}

fn resolve_runtime_root(
    app_handle: &tauri::AppHandle,
    workspace_root: &Path,
) -> Result<PathBuf, String> {
    if has_workspace_venv(workspace_root) {
        return Ok(workspace_root.to_path_buf());
    }

    if directory_is_writable(workspace_root) {
        return Ok(workspace_root.to_path_buf());
    }

    let app_data_dir = app_handle.path().app_local_data_dir().map_err(|error| {
        format!("Cannot resolve per-user runtime directory for packaged desktop mode: {error}")
    })?;
    let runtime_root = app_data_dir.join("runtime");
    if directory_is_writable(&runtime_root) {
        return Ok(runtime_root);
    }

    Err(format!(
        "Cannot access writable runtime directory at {}.",
        runtime_root.display()
    ))
}

fn configure_background_command(command: &mut Command) -> &mut Command {
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

fn run_command_with_timeout(
    command: &mut Command,
    timeout: Duration,
    context: &str,
) -> Result<bool, String> {
    let mut child = command
        .spawn()
        .map_err(|error| format!("Failed to spawn {context}: {error}"))?;
    let start = Instant::now();

    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("Failed while waiting for {context}: {error}"))?
        {
            return Ok(status.success());
        }

        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "{context} timed out after {} seconds.",
                timeout.as_secs()
            ));
        }

        thread::sleep(Duration::from_millis(250));
    }
}

fn spawn_backend(app_handle: &tauri::AppHandle, state: &BackendChildState) -> Result<(), String> {
    #[cfg(not(target_os = "windows"))]
    {
        let _ = app_handle;
        let _ = state;
        return Err(String::from(
            "Local Mode (v2) is currently supported on Windows only.",
        ));
    }

    #[cfg(target_os = "windows")]
    {
        let workspace_root = find_workspace_root(app_handle).ok_or_else(|| {
            String::from(
                "Cannot resolve packaged backend workspace (missing pyproject.toml/XREPORT).",
            )
        })?;
        let runtime_root = resolve_runtime_root(app_handle, &workspace_root)?;
        let project_dir = workspace_root.join("XREPORT");
        let env_path = project_dir.join("settings").join(".env");
        let backend_config = resolve_backend_launch_config(&env_path);
        let uv_exe = project_dir
            .join("resources")
            .join("runtimes")
            .join("uv")
            .join("uv.exe");
        let python_exe = project_dir
            .join("resources")
            .join("runtimes")
            .join("python")
            .join("python.exe");
        let venv_dir = runtime_root.join(".venv");
        let venv_python_exe = venv_dir.join("Scripts").join("python.exe");
        let uv_cache_dir = runtime_root.join(".uv-cache");

        if !uv_exe.is_file() {
            return Err(format!(
                "Bundled uv runtime not found at {}",
                uv_exe.display()
            ));
        }
        if !python_exe.is_file() {
            return Err(format!(
                "Bundled python runtime not found at {}",
                python_exe.display()
            ));
        }

        let python_exe_str = python_exe.to_string_lossy().to_string();
        let uv_cache_dir_str = uv_cache_dir.to_string_lossy().to_string();
        let venv_dir_str = venv_dir.to_string_lossy().to_string();
        let mut sync_args = vec![String::from("sync")];
        if backend_config.install_extras {
            sync_args.push(String::from("--all-extras"));
        }
        sync_args.push(String::from("--frozen"));

        let mut sync_with_embedded_args = vec![
            String::from("sync"),
            String::from("--python"),
            python_exe_str.clone(),
        ];
        if backend_config.install_extras {
            sync_with_embedded_args.push(String::from("--all-extras"));
        }
        sync_with_embedded_args.push(String::from("--frozen"));

        if !venv_python_exe.is_file() {
            fs::create_dir_all(&uv_cache_dir).map_err(|error| {
                format!(
                    "Cannot create uv cache directory at {}: {error}",
                    uv_cache_dir.display()
                )
            })?;

            render_startup_status(
                app_handle,
                "Synchronizing Python environment. First launch can take several minutes while dependencies are prepared.",
            );

            let mut embedded_sync_command = Command::new(&uv_exe);
            configure_background_command(&mut embedded_sync_command);
            embedded_sync_command
                .args(sync_with_embedded_args.iter().map(|s| s.as_str()))
                .current_dir(&workspace_root)
                .env("UV_PROJECT_ENVIRONMENT", &venv_dir_str)
                .env("UV_CACHE_DIR", &uv_cache_dir_str)
                .stdout(Stdio::null())
                .stderr(Stdio::null());

            let embedded_sync_ok = run_command_with_timeout(
                &mut embedded_sync_command,
                Duration::from_secs(15 * 60),
                "uv sync (embedded python)",
            )?;

            if !embedded_sync_ok {
                let mut fallback_sync_command = Command::new(&uv_exe);
                configure_background_command(&mut fallback_sync_command);
                fallback_sync_command
                    .args(sync_args.iter().map(|s| s.as_str()))
                    .current_dir(&workspace_root)
                    .env("UV_PROJECT_ENVIRONMENT", &venv_dir_str)
                    .env("UV_CACHE_DIR", &uv_cache_dir_str)
                    .stdout(Stdio::null())
                    .stderr(Stdio::null());

                let fallback_sync_ok = run_command_with_timeout(
                    &mut fallback_sync_command,
                    Duration::from_secs(15 * 60),
                    "uv sync fallback",
                )?;

                if !fallback_sync_ok {
                    return Err(String::from("uv sync failed for packaged runtime."));
                }
            }

            if !venv_python_exe.is_file() {
                return Err(format!(
                    "Python environment setup completed but {} is missing.",
                    venv_python_exe.display()
                ));
            }
        }

        render_startup_status(app_handle, "Starting local API service.");

        let backend_host = backend_config.host.clone();
        let backend_port = backend_config.port;
        let backend_port_str = backend_port.to_string();
        let mut child_command = Command::new(&venv_python_exe);
        configure_background_command(&mut child_command);
        child_command.arg("-m").arg("uvicorn");
        child_command
            .arg("XREPORT.server.app:app")
            .arg("--host")
            .arg(&backend_host)
            .arg("--port")
            .arg(&backend_port_str)
            .arg("--log-level")
            .arg("info");
        if backend_config.reload {
            child_command.arg("--reload");
        }

        let child = child_command
            .current_dir(&workspace_root)
            .env("XREPORT_TAURI_MODE", "true")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("Failed to start packaged backend process: {error}"))?;

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
                "Timed out waiting for backend at http://{backend_host}:{backend_port}/.\n\nThe packaged backend may still be installing Python dependencies. If this is the first launch, wait a little longer and try again."
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
                let mut taskkill = Command::new("taskkill");
                let _ = configure_background_command(&mut taskkill)
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

            render_startup_status(
                app.handle(),
                "Preparing local backend. First launch can take a minute while Python dependencies are synchronized.\n\nThis window will switch automatically once the API is ready.",
            );

            let app_handle = app.handle().clone();
            let state = state.inner().clone();
            thread::spawn(move || {
                if let Err(error) = spawn_backend(&app_handle, &state) {
                    render_startup_error(&app_handle, &error);
                }
            });

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
