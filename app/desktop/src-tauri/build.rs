use std::collections::HashSet;
use std::fs::File;

fn validate_release_inputs(archive_path: &std::path::Path, ui: &std::path::Path) {
    let file = File::open(archive_path).unwrap_or_else(|error| {
        panic!(
            "Desktop runtime has not been generated. Run the desktop preparation step before `tauri build` ({error})."
        )
    });
    let mut archive = zip::ZipArchive::new(file).unwrap_or_else(|error| {
        panic!(
            "Desktop runtime archive is missing or invalid at {}: {error}",
            archive_path.display()
        )
    });
    if !ui.join("index.html").is_file() || !ui.join("error.html").is_file() {
        panic!(
            "Desktop UI has not been generated at {}. Run the desktop preparation step.",
            ui.display()
        );
    }
    let required = [
        "client/index.html",
        "client/error.html",
        "backend/xreport-backend.exe",
        "settings/.env.example",
        "settings/configurations.json",
        "settings/inference_models.json",
        "runtime-manifest.json",
    ];
    let mut names = HashSet::new();
    for index in 0..archive.len() {
        let entry = archive.by_index(index).unwrap_or_else(|error| {
            panic!("Desktop runtime archive entry {index} is invalid: {error}")
        });
        if entry.is_dir() {
            panic!(
                "Desktop runtime archive contains a directory entry: {}",
                entry.name()
            );
        }
        if !names.insert(entry.name().to_ascii_lowercase()) {
            panic!(
                "Desktop runtime archive contains a duplicate entry: {}",
                entry.name()
            );
        }
    }
    for member in required {
        if !names.contains(member) {
            panic!(
                "Desktop runtime archive is missing required member {member}. Run the desktop preparation step."
            );
        }
    }

    let manifest = {
        let mut entry = archive
            .by_name("runtime-manifest.json")
            .expect("runtime manifest was checked above");
        serde_json::from_reader::<_, serde_json::Value>(&mut entry)
            .unwrap_or_else(|error| panic!("Desktop runtime manifest is invalid: {error}"))
    };
    let expected_variant =
        std::env::var("XREPORT_DESKTOP_VARIANT").unwrap_or_else(|_| "cpu".to_string());
    let expected_version =
        std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".to_string());
    if manifest.get("format").and_then(serde_json::Value::as_u64) != Some(2)
        || manifest
            .get("application")
            .and_then(serde_json::Value::as_str)
            != Some("XREPORT")
        || manifest.get("version").and_then(serde_json::Value::as_str) != Some(&expected_version)
        || manifest.get("variant").and_then(serde_json::Value::as_str) != Some(&expected_variant)
        || manifest
            .get("architecture")
            .and_then(serde_json::Value::as_str)
            != Some("windows-x64")
        || manifest
            .get("backend_executable")
            .and_then(serde_json::Value::as_str)
            != Some("backend/XREPORT-backend.exe")
    {
        panic!("Desktop runtime manifest does not match the release shell.");
    }
    let source_commit = manifest
        .get("source_commit")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let payload_sha256 = manifest
        .get("payload_sha256")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if source_commit.len() != 40
        || !source_commit
            .chars()
            .all(|character| character.is_ascii_hexdigit())
        || payload_sha256.len() != 64
        || !payload_sha256
            .chars()
            .all(|character| character.is_ascii_hexdigit())
        || manifest
            .get("created_utc")
            .and_then(serde_json::Value::as_str)
            .is_none_or(|value| value.trim().is_empty())
        || manifest
            .get("file_count")
            .and_then(serde_json::Value::as_u64)
            .is_none()
        || manifest
            .get("payload_bytes")
            .and_then(serde_json::Value::as_u64)
            .is_none()
    {
        panic!("Desktop runtime manifest has invalid release metadata.");
    }
}

fn main() {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let archive = manifest_dir.join("generated").join("runtime.zip");
    let ui = manifest_dir.join("ui");
    let profile = std::env::var("PROFILE").unwrap_or_default();
    if profile == "release" {
        validate_release_inputs(&archive, &ui);
    }
    // `tauri_build` validates bundle resources even for a debug/test compile.
    // Debug tests intentionally run without generated release assets; the
    // release path above remains fail-closed and is prepared by the launcher.
    if profile == "release" {
        tauri_build::build();
    }
    let variant = std::env::var("XREPORT_DESKTOP_VARIANT").unwrap_or_else(|_| "cpu".to_string());
    println!("cargo:rustc-env=XREPORT_DESKTOP_VARIANT={variant}");
    println!("cargo:rerun-if-changed=generated/runtime.zip");
    println!("cargo:rerun-if-changed=ui");
    println!("cargo:rerun-if-env-changed=XREPORT_DESKTOP_VARIANT");
}
