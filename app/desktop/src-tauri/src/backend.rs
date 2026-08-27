use crate::runtime::extract_runtime;
use crate::windows_job::JobObject;
use base64::Engine;
use serde::Deserialize;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Debug, Deserialize)]
struct ReadyContract {
    host: String,
    port: u16,
    pid: u32,
    version: String,
    variant: String,
}

pub struct BackendHandle {
    child: Mutex<Child>,
    stopped: AtomicBool,
    _job: Option<JobObject>,
    pub data_root: PathBuf,
    pub ready_file: PathBuf,
    pub session_file: PathBuf,
    pub token: String,
    pub port: u16,
}

unsafe impl Send for BackendHandle {}
unsafe impl Sync for BackendHandle {}

fn shell_log(data_root: &Path, message: &str) {
    let directory = data_root.join("logs");
    if fs::create_dir_all(&directory).is_err() {
        return;
    }
    let path = directory.join("desktop-shell.log");
    if path
        .metadata()
        .map(|metadata| metadata.len() > 2 * 1024 * 1024)
        .unwrap_or(false)
    {
        let first = directory.join("desktop-shell.log.1");
        let second = directory.join("desktop-shell.log.2");
        let _ = fs::remove_file(&second);
        let _ = fs::rename(&first, &second);
        let _ = fs::rename(&path, &first);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{}", message.replace('\n', " "));
    }
}

fn backend_log(data_root: &Path) -> Result<std::fs::File, String> {
    let directory = data_root.join("logs");
    fs::create_dir_all(&directory)
        .map_err(|error| format!("create backend log directory: {error}"))?;
    let path = directory.join("backend.log");
    if path
        .metadata()
        .map(|metadata| metadata.len() > 8 * 1024 * 1024)
        .unwrap_or(false)
    {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("read backend log clock: {error}"))?
            .as_secs();
        let rotated = directory.join(format!("backend-{timestamp}.log"));
        fs::rename(&path, rotated).map_err(|error| format!("rotate backend log: {error}"))?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("open backend log: {error}"))
}

fn create_token() -> String {
    let first = Uuid::new_v4().as_bytes().to_owned();
    let second = Uuid::new_v4().as_bytes().to_owned();
    let mut bytes = [0u8; 32];
    bytes[..16].copy_from_slice(&first);
    bytes[16..].copy_from_slice(&second);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn request(port: u16, method: &str, path: &str, token: &str) -> Result<u16, String> {
    let mut stream =
        TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(2))
            .map_err(|error| error.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| error.to_string())?;
    let payload = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-XREPORT-Desktop-Token: {token}\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(payload.as_bytes())
        .map_err(|error| error.to_string())?;
    let mut response = [0u8; 512];
    let count = stream
        .read(&mut response)
        .map_err(|error| error.to_string())?;
    let status = String::from_utf8_lossy(&response[..count]);
    status
        .split_whitespace()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| "invalid backend HTTP response".to_string())
}

fn health_is_ready(port: u16, token: &str) -> bool {
    request(port, "GET", "/api/health", token).is_ok_and(|status| status == 200)
}

fn read_ready(path: &Path, version: &str, variant: &str) -> Result<Option<ReadyContract>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let payload =
        fs::read(path).map_err(|error| format!("read backend readiness contract: {error}"))?;
    let ready: ReadyContract = serde_json::from_slice(&payload)
        .map_err(|error| format!("parse backend readiness contract: {error}"))?;
    if ready.version != version || ready.variant != variant || ready.host != "127.0.0.1" {
        return Err("backend readiness contract does not match the desktop shell".to_string());
    }
    if ready.pid == 0 || ready.port == 0 {
        return Err("backend readiness contract contains an invalid process or port".to_string());
    }
    Ok(Some(ready))
}

fn configure_process(command: &mut Command, data_root: &Path) -> Result<(), String> {
    let nltk_root = data_root.join("nltk");
    let matplotlib_root = data_root.join("caches").join("matplotlib");
    fs::create_dir_all(&nltk_root)
        .map_err(|error| format!("create NLTK data directory: {error}"))?;
    fs::create_dir_all(&matplotlib_root)
        .map_err(|error| format!("create Matplotlib cache directory: {error}"))?;
    command
        .env_clear()
        .env("PATH", std::env::var("PATH").unwrap_or_default())
        .env(
            "SYSTEMROOT",
            std::env::var("SYSTEMROOT").unwrap_or_default(),
        )
        .env("WINDIR", std::env::var("WINDIR").unwrap_or_default())
        .env("TEMP", std::env::var("TEMP").unwrap_or_default())
        .env("TMP", std::env::var("TMP").unwrap_or_default())
        .env("USERNAME", std::env::var("USERNAME").unwrap_or_default())
        .env(
            "USERPROFILE",
            std::env::var("USERPROFILE").unwrap_or_default(),
        )
        .env("APPDATA", std::env::var("APPDATA").unwrap_or_default())
        .env(
            "LOCALAPPDATA",
            std::env::var("LOCALAPPDATA").unwrap_or_default(),
        )
        .env("CUDA_PATH", std::env::var("CUDA_PATH").unwrap_or_default())
        .env(
            "CUDA_PATH_V13_0",
            std::env::var("CUDA_PATH_V13_0").unwrap_or_default(),
        )
        // PyInstaller runtime hooks import NLTK/Matplotlib before the Python
        // entry point can configure its environment.  Set their writable
        // user-data locations before spawning the frozen process.
        .env("NLTK_DATA", &nltk_root)
        .env("MPLCONFIGDIR", &matplotlib_root);
    Ok(())
}

pub fn start_backend(
    data_root: &Path,
    version: &str,
    variant: &str,
) -> Result<BackendHandle, String> {
    let startup_started = Instant::now();
    let runtime_started = Instant::now();
    let runtime = extract_runtime(data_root, version, variant)?;
    shell_log(
        data_root,
        &format!(
            "startup phase=runtime_ready cache_hit={} phase_elapsed_ms={:.0} total_elapsed_ms={:.0}",
            runtime.cache_hit,
            runtime_started.elapsed().as_secs_f64() * 1000.0,
            startup_started.elapsed().as_secs_f64() * 1000.0
        ),
    );
    let state_dir = data_root.join("state");
    fs::create_dir_all(&state_dir)
        .map_err(|error| format!("create desktop state directory: {error}"))?;
    let ready_file = state_dir.join("desktop-ready.json");
    let session_file = state_dir.join("desktop-session.json");
    let _ = fs::remove_file(&ready_file);
    let _ = fs::remove_file(&session_file);
    let token = create_token();

    shell_log(
        data_root,
        &format!(
            "startup phase=backend_launching variant={variant} version={version} total_elapsed_ms={:.0}",
            startup_started.elapsed().as_secs_f64() * 1000.0
        ),
    );
    let backend_log = backend_log(data_root)?;
    let backend_log_error = backend_log
        .try_clone()
        .map_err(|error| format!("clone backend log handle: {error}"))?;
    let mut command = Command::new(&runtime.backend_executable);
    command
        .current_dir(&runtime.root)
        .args([
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            ready_file.to_string_lossy().as_ref(),
            "--session-file",
            session_file.to_string_lossy().as_ref(),
            "--variant",
            variant,
            "--version",
            version,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::from(backend_log))
        .stderr(Stdio::from(backend_log_error));
    configure_process(&mut command, data_root)?;
    command
        .env("XREPORT_DESKTOP", "true")
        .env("XREPORT_RUNTIME_ROOT", &runtime.root)
        .env("XREPORT_DATA_ROOT", data_root)
        .env("XREPORT_RELEASE_VERSION", version)
        .env("XREPORT_RUNTIME_VARIANT", variant)
        .env("XREPORT_DESKTOP_TOKEN", &token)
        .env("XREPORT_CLIENT_DIST_DIR", &runtime.client_dir)
        .env("PYTHONUNBUFFERED", "1");
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut command, 0x08000000);

    let mut child = command
        .spawn()
        .map_err(|error| format!("start packaged backend: {error}"))?;
    shell_log(
        data_root,
        &format!(
            "startup phase=backend_spawned pid={} total_elapsed_ms={:.0}",
            child.id(),
            startup_started.elapsed().as_secs_f64() * 1000.0
        ),
    );
    let job = match JobObject::attach(child.id()) {
        Ok(job) => job,
        Err(error) => {
            let _ = child.kill();
            return Err(format!("attach backend to Windows job object: {error}"));
        }
    };
    let deadline = Instant::now() + Duration::from_secs(120);
    let contract = loop {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!("packaged backend exited during startup ({status})"));
        }
        if let Some(ready) = read_ready(&ready_file, version, variant)? {
            if health_is_ready(ready.port, &token) {
                break ready;
            }
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            return Err("packaged backend did not become ready within 120 seconds".to_string());
        }
        thread::sleep(Duration::from_millis(250));
    };
    shell_log(
        data_root,
        &format!(
            "startup phase=backend_ready port={} total_elapsed_ms={:.0}",
            contract.port,
            startup_started.elapsed().as_secs_f64() * 1000.0
        ),
    );
    Ok(BackendHandle {
        child: Mutex::new(child),
        stopped: AtomicBool::new(false),
        _job: Some(job),
        data_root: data_root.to_path_buf(),
        ready_file,
        session_file,
        token,
        port: contract.port,
    })
}

impl BackendHandle {
    pub fn bootstrap_url(&self) -> String {
        format!(
            "http://127.0.0.1:{}/__xreport/bootstrap?token={}",
            self.port, self.token
        )
    }

    pub fn has_exited(&self) -> Result<Option<i32>, String> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| "backend process lock poisoned".to_string())?;
        child
            .try_wait()
            .map(|status| status.and_then(|value| value.code()))
            .map_err(|error| format!("check backend process: {error}"))
    }

    pub fn stop(&self) {
        if self.stopped.swap(true, Ordering::AcqRel) {
            return;
        }
        let shutdown_started = Instant::now();
        let _ = request(self.port, "POST", "/__xreport/shutdown", &self.token);
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            let exited = self.has_exited().ok().flatten().is_some();
            if exited || Instant::now() >= deadline {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
        if let Ok(mut child) = self.child.lock() {
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        let _ = fs::remove_file(&self.ready_file);
        let _ = fs::remove_file(&self.session_file);
        shell_log(
            &self.data_root,
            &format!(
                "backend stopped elapsed_ms={:.0}",
                shutdown_started.elapsed().as_secs_f64() * 1000.0
            ),
        );
    }
}

impl Drop for BackendHandle {
    fn drop(&mut self) {
        self.stop();
    }
}
