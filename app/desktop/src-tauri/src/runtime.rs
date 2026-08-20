use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;
use zip::ZipArchive;

const OVERLAY_MAGIC: &[u8; 8] = b"XRPZIP01";
const OVERLAY_FOOTER_LEN: u64 = 8 + 8 + 8;

/// A seekable view over a region of a file.
///
/// CUDA runtime archives are several gigabytes.  Keeping the archive as a
/// file/resource (or as an appended PE overlay for the portable build) lets
/// ZIP and extraction remain streaming operations instead of asking rustc or
/// the shell to allocate the complete archive in memory.
struct BoundedFile {
    file: File,
    start: u64,
    length: u64,
    position: u64,
}

impl BoundedFile {
    fn new(file: File, start: u64, length: u64) -> Self {
        Self {
            file,
            start,
            length,
            position: 0,
        }
    }
}

impl Read for BoundedFile {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.position >= self.length || buffer.is_empty() {
            return Ok(0);
        }
        let remaining = self.length - self.position;
        let requested = remaining.min(buffer.len() as u64) as usize;
        self.file
            .seek(SeekFrom::Start(self.start + self.position))?;
        let count = self.file.read(&mut buffer[..requested])?;
        self.position += count as u64;
        Ok(count)
    }
}

impl Seek for BoundedFile {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let target = match position {
            SeekFrom::Start(value) => i128::from(value),
            SeekFrom::Current(value) => i128::from(self.position) + i128::from(value),
            SeekFrom::End(value) => i128::from(self.length) + i128::from(value),
        };
        if !(0..=i128::from(self.length)).contains(&target) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "runtime archive seek outside its bounded region",
            ));
        }
        self.position = target as u64;
        Ok(self.position)
    }
}

fn archive_from_file(path: &Path) -> Result<ZipArchive<BoundedFile>, String> {
    let file =
        File::open(path).map_err(|error| format!("open runtime archive {path:?}: {error}"))?;
    let length = file
        .metadata()
        .map_err(|error| format!("read runtime archive metadata: {error}"))?
        .len();
    ZipArchive::new(BoundedFile::new(file, 0, length))
        .map_err(|error| format!("open runtime archive {path:?}: {error}"))
}

fn archive_from_overlay(path: &Path) -> Result<Option<ZipArchive<BoundedFile>>, String> {
    let mut file =
        File::open(path).map_err(|error| format!("open portable executable: {error}"))?;
    let length = file
        .metadata()
        .map_err(|error| format!("read portable executable metadata: {error}"))?
        .len();
    if length < OVERLAY_FOOTER_LEN {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(length - OVERLAY_FOOTER_LEN))
        .map_err(|error| format!("seek portable runtime footer: {error}"))?;
    let mut footer = [0u8; OVERLAY_FOOTER_LEN as usize];
    file.read_exact(&mut footer)
        .map_err(|error| format!("read portable runtime footer: {error}"))?;
    if &footer[..8] != OVERLAY_MAGIC {
        return Ok(None);
    }
    let offset = u64::from_le_bytes(footer[8..16].try_into().expect("fixed footer slice"));
    let archive_length = u64::from_le_bytes(footer[16..24].try_into().expect("fixed footer slice"));
    if offset > length - OVERLAY_FOOTER_LEN || archive_length > length - OVERLAY_FOOTER_LEN - offset
    {
        return Err("portable runtime overlay bounds are invalid".to_string());
    }
    ZipArchive::new(BoundedFile::new(file, offset, archive_length))
        .map(Some)
        .map_err(|error| format!("open portable runtime overlay: {error}"))
}

fn open_runtime_archive(
    expected_variant: &str,
    expected_version: &str,
) -> Result<ZipArchive<BoundedFile>, String> {
    let executable =
        env::current_exe().map_err(|error| format!("locate desktop executable: {error}"))?;
    let executable_dir = executable
        .parent()
        .ok_or_else(|| "desktop executable has no parent directory".to_string())?;
    // Prefer the authenticated single-file overlay.  If its footer is
    // present but corrupt, fail closed instead of silently accepting a
    // replaceable archive dropped beside the executable.
    if let Some(archive) = archive_from_overlay(&executable)? {
        return Ok(archive);
    }
    let sibling_names = [
        "runtime.zip".to_string(),
        format!("runtime-{expected_variant}-{expected_version}.zip"),
    ];
    for name in sibling_names {
        let candidate = executable_dir.join(name);
        if candidate.is_file() {
            return archive_from_file(&candidate);
        }
    }
    // Tauri's Windows resource bundler preserves the resource key's parent
    // directory (currently `generated`), while development/manual layouts
    // may use `resources`.  Check both so MSI and portable layouts share the
    // same fail-closed extraction path.
    for resource_archive in [
        executable_dir.join("generated").join("runtime.zip"),
        executable_dir.join("resources").join("runtime.zip"),
    ] {
        if resource_archive.is_file() {
            return archive_from_file(&resource_archive);
        }
    }
    Err("no XREPORT runtime archive was found beside the desktop executable".to_string())
}

#[derive(Debug, Deserialize)]
struct RuntimeManifest {
    version: String,
    variant: String,
    payload_sha256: String,
    #[serde(default = "default_backend")]
    backend_executable: String,
}

fn default_backend() -> String {
    "backend/XREPORT-backend.exe".to_string()
}

#[derive(Clone, Debug)]
pub struct RuntimeInfo {
    pub root: PathBuf,
    pub backend_executable: PathBuf,
    pub client_dir: PathBuf,
    pub payload_sha256: String,
}

fn safe_member(name: &str) -> Result<PathBuf, String> {
    let path = Path::new(name);
    if name.is_empty() || name.contains('\\') || path.is_absolute() {
        return Err(format!("unsafe runtime archive member: {name}"));
    }
    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(format!("unsafe runtime archive member: {name}"));
        }
    }
    let lower = name.to_ascii_lowercase();
    if lower == ".env"
        || lower == "database.db"
        || lower.ends_with("/.env")
        || lower.ends_with("/database.db")
        || lower.ends_with(".sqlite")
        || lower.ends_with(".sqlite3")
        || lower.ends_with(".log")
        || lower.contains("/__pycache__/")
        || lower.contains("/node_modules/")
    {
        return Err(format!("forbidden runtime archive member: {name}"));
    }
    Ok(path.to_path_buf())
}

fn validate_manifest(
    root: &Path,
    expected_version: &str,
    expected_variant: &str,
) -> Result<RuntimeInfo, String> {
    let manifest_path = root.join("runtime-manifest.json");
    let manifest: RuntimeManifest = serde_json::from_slice(
        &fs::read(&manifest_path).map_err(|error| format!("read runtime manifest: {error}"))?,
    )
    .map_err(|error| format!("parse runtime manifest: {error}"))?;
    if manifest.version != expected_version || manifest.variant != expected_variant {
        return Err("runtime manifest does not match the desktop shell".to_string());
    }
    if manifest.payload_sha256.len() != 64
        || !manifest
            .payload_sha256
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return Err("runtime manifest contains an invalid payload hash".to_string());
    }
    let backend = root.join(&manifest.backend_executable);
    let client = root.join("client");
    if !backend.is_file() || !client.join("index.html").is_file() {
        return Err("runtime is missing the frozen backend or Angular client".to_string());
    }
    Ok(RuntimeInfo {
        root: root.to_path_buf(),
        backend_executable: backend,
        client_dir: client,
        payload_sha256: manifest.payload_sha256,
    })
}

fn existing_runtime(
    target: &Path,
    expected_version: &str,
    expected_variant: &str,
    digest: &str,
) -> Result<Option<RuntimeInfo>, String> {
    if !target.is_dir() {
        return Ok(None);
    }
    match validate_manifest(target, expected_version, expected_variant) {
        Ok(info) if info.payload_sha256.eq_ignore_ascii_case(digest) => Ok(Some(info)),
        Ok(_) | Err(_) => Ok(None),
    }
}

pub fn extract_runtime(
    data_root: &Path,
    expected_version: &str,
    expected_variant: &str,
) -> Result<RuntimeInfo, String> {
    let mut archive = open_runtime_archive(expected_variant, expected_version)?;
    let runtime_parent = data_root
        .parent()
        .ok_or_else(|| "desktop data root has no application parent".to_string())?
        .join("runtime")
        .join(expected_variant)
        .join(expected_version);
    fs::create_dir_all(&runtime_parent)
        .map_err(|error| format!("create runtime directory: {error}"))?;
    let staging = runtime_parent.join(format!(".staging-{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&staging)
        .map_err(|error| format!("create runtime staging directory: {error}"))?;

    let result = (|| -> Result<RuntimeInfo, String> {
        let mut digest = Sha256::new();
        let mut members = HashSet::new();
        for index in 0..archive.len() {
            let mut entry = archive
                .by_index(index)
                .map_err(|error| format!("read runtime archive entry {index}: {error}"))?;
            let member_name = entry.name().to_string();
            let relative = safe_member(&member_name)?;
            if !members.insert(member_name.clone()) {
                return Err(format!("duplicate runtime archive member: {member_name}"));
            }
            if entry
                .unix_mode()
                .is_some_and(|mode| mode & 0o170000 == 0o120000)
            {
                return Err(format!("symlink runtime archive member: {member_name}"));
            }
            if entry.is_dir() {
                continue;
            }
            let destination = staging.join(&relative);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("create runtime member directory: {error}"))?;
            }
            let mut output = File::create(&destination)
                .map_err(|error| format!("create runtime member {member_name}: {error}"))?;
            if member_name != "runtime-manifest.json" {
                digest.update(member_name.as_bytes());
                digest.update([0]);
            }
            let mut buffer = [0u8; 1024 * 1024];
            loop {
                let count = entry
                    .read(&mut buffer)
                    .map_err(|error| format!("read runtime member {member_name}: {error}"))?;
                if count == 0 {
                    break;
                }
                if member_name != "runtime-manifest.json" {
                    digest.update(&buffer[..count]);
                }
                output
                    .write_all(&buffer[..count])
                    .map_err(|error| format!("write runtime member {member_name}: {error}"))?;
            }
        }
        let payload_hash = hex::encode(digest.finalize());
        let info = validate_manifest(&staging, expected_version, expected_variant)?;
        if !info.payload_sha256.eq_ignore_ascii_case(&payload_hash) {
            return Err("embedded runtime payload hash does not match its manifest".to_string());
        }
        let target = runtime_parent.join(&payload_hash);
        if let Some(existing) =
            existing_runtime(&target, expected_version, expected_variant, &payload_hash)?
        {
            return Ok(existing);
        }
        if target.exists() {
            fs::remove_dir_all(&target)
                .map_err(|error| format!("replace invalid runtime: {error}"))?;
        }
        fs::rename(&staging, &target)
            .map_err(|error| format!("commit extracted runtime: {error}"))?;
        validate_manifest(&target, expected_version, expected_variant)
    })();

    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::safe_member;

    #[test]
    fn archive_member_validation_rejects_escape_and_mutable_files() {
        assert!(safe_member("client/index.html").is_ok());
        assert!(safe_member("../database.db").is_err());
        assert!(safe_member("database.db").is_err());
        assert!(safe_member("settings/.env").is_err());
        assert!(safe_member("backend\\child.exe").is_err());
    }
}
