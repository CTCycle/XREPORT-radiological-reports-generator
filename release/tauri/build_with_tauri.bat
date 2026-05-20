@echo off
setlocal enabledelayedexpansion

set "script_dir=%~dp0"
for %%I in ("%script_dir%..\..") do set "repo_root=%%~fI"
set "app_dir=%repo_root%\app"
set "client_dir=%app_dir%\client"
set "tauri_dir=%client_dir%\src-tauri"
set "bundle_source_dir=%tauri_dir%\r"
set "bundle_dir=%tauri_dir%\target\release\bundle"
set "release_export_dir=%repo_root%\release\windows"

set "runtime_python_exe=%repo_root%\runtimes\python\python.exe"
set "runtime_uv_exe=%repo_root%\runtimes\uv\uv.exe"
set "runtime_uv_lock=%repo_root%\app\server\uv.lock"
set "runtime_node_dir=%repo_root%\runtimes\nodejs"
set "node_cmd=%runtime_node_dir%\node.exe"
set "npm_cmd=%runtime_node_dir%\npm.cmd"
set "runtime_database_file=%app_dir%\resources\database.db"

echo [TAURI] Release build helper

echo [CHECK] Validating bundled runtimes...
call :require_file "%runtime_python_exe%" "embedded Python runtime" || goto build_error
call :require_file "%runtime_uv_exe%" "embedded uv runtime" || goto build_error
call :require_file "%runtime_uv_lock%" "backend uv lockfile" || goto build_error
call :require_file "%node_cmd%" "embedded Node.js runtime" || goto build_error
call :require_file "%npm_cmd%" "embedded npm runtime" || goto build_error
call :require_file "%app_dir%\server\pyproject.toml" "backend pyproject.toml" || goto build_error
call :require_file "%runtime_database_file%" "runtime sqlite database" || goto build_error

echo [CHECK] Preparing Tauri bundle sources...
call :prepare_bundle_sources || goto build_error

echo [CHECK] Resolving Cargo...
set "cargo_cmd="
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "cargo_cmd=%USERPROFILE%\.cargo\bin\cargo.exe"
if not defined cargo_cmd (
  cargo --version >nul 2>&1
  if not errorlevel 1 set "cargo_cmd=cargo"
)
if not defined cargo_cmd (
  echo [FATAL] Rust/Cargo not found. Install Rust first: https://rustup.rs/
  goto build_error
)

call :check_rust_toolchain "%cargo_cmd%" || goto build_error
for /f "delims=" %%V in ('"%cargo_cmd%" --version 2^>nul') do set "cargo_version=%%V"
echo [INFO] Cargo command: %cargo_cmd%
if defined cargo_version echo [INFO] !cargo_version!
if /I not "%cargo_cmd%"=="cargo" (
  for %%I in ("%cargo_cmd%") do set "PATH=%%~dpI;%PATH%"
)
set "CARGO=%cargo_cmd%"

if /I not "%node_cmd%"=="node" (
  for %%I in ("%node_cmd%") do set "PATH=%%~dpI;%PATH%"
)

for /f "delims=" %%V in ('"%node_cmd%" --version 2^>nul') do set "node_version=%%V"
for /f "delims=" %%V in ('"%npm_cmd%" --version 2^>nul') do set "npm_version=%%V"
echo [INFO] npm command: %npm_cmd%
echo [INFO] node command: %node_cmd%
if defined node_version echo [INFO] Node.js version: !node_version!
if defined npm_version echo [INFO] npm version: !npm_version!

if not exist "%client_dir%\package.json" (
  echo [FATAL] Missing client package.json at "%client_dir%"
  goto build_error
)

set "RUST_BACKTRACE=1"
set "CARGO_TERM_PROGRESS_WHEN=auto"

echo [STEP 1/2] Installing frontend dependencies
pushd "%client_dir%" >nul
if exist "package-lock.json" (
  echo [CMD] "%npm_cmd%" ci --foreground-scripts
  call "%npm_cmd%" ci --foreground-scripts
) else (
  echo [CMD] "%npm_cmd%" install --foreground-scripts
  call "%npm_cmd%" install --foreground-scripts
)
if errorlevel 1 (
  popd >nul
  echo [FATAL] npm dependency installation failed.
  goto build_error
)

echo [CHECK] Refreshing Tauri bundle sources...
call :prepare_bundle_sources || (
  popd >nul
  goto build_error
)

echo [STEP 2/2] Building Tauri application
echo [CMD] "%npm_cmd%" run tauri:build:release
call "%npm_cmd%" run tauri:build:release
if errorlevel 1 (
  popd >nul
  echo [FATAL] Tauri build failed.
  goto build_error
)
popd >nul

call :cleanup_bundle_sources

echo [OK] Build completed successfully.
if exist "%release_export_dir%" (
  echo [INFO] User-facing release artifacts:
  echo        %release_export_dir%
) else if exist "%bundle_dir%" (
  echo [INFO] Release artifacts:
  echo        %bundle_dir%
) else (
  echo [WARN] Build finished but release directories were not found.
  echo        %release_export_dir%
  echo        %bundle_dir%
)

endlocal & exit /b 0

:require_file
if exist "%~1" (
  echo [OK] %~2 found: %~1
  exit /b 0
)
echo [FATAL] Missing %~2 at "%~1"
echo         Run start_on_windows.bat first to install portable runtimes.
exit /b 1

:prepare_bundle_sources
call :cleanup_bundle_sources

md "%bundle_source_dir%" >nul 2>&1
if errorlevel 1 (
  echo [FATAL] Failed to create bundle source directory "%bundle_source_dir%".
  exit /b 1
)
md "%bundle_source_dir%\app" >nul 2>&1
md "%bundle_source_dir%\app\client" >nul 2>&1
md "%bundle_source_dir%\runtimes" >nul 2>&1

copy /y "%app_dir%\server\pyproject.toml" "%bundle_source_dir%\pyproject.toml" >nul
if errorlevel 1 (
  echo [FATAL] Failed to stage pyproject.toml for Tauri bundling.
  exit /b 1
)
copy /y "%runtime_uv_lock%" "%bundle_source_dir%\uv.lock" >nul
if errorlevel 1 (
  echo [FATAL] Failed to stage uv.lock for Tauri bundling.
  exit /b 1
)

if not exist "%client_dir%\dist" md "%client_dir%\dist" >nul 2>&1

call :make_junction "%bundle_source_dir%\app\server" "%app_dir%\server" || exit /b 1
call :make_junction "%bundle_source_dir%\app\scripts" "%app_dir%\scripts" || exit /b 1
call :make_junction "%bundle_source_dir%\settings" "%repo_root%\settings" || exit /b 1
call :make_junction "%bundle_source_dir%\app\client\dist" "%client_dir%\dist" || exit /b 1
call :make_junction "%bundle_source_dir%\app\resources" "%app_dir%\resources" || exit /b 1
call :make_junction "%bundle_source_dir%\runtimes\python" "%repo_root%\runtimes\python" || exit /b 1
call :make_junction "%bundle_source_dir%\runtimes\uv" "%repo_root%\runtimes\uv" || exit /b 1
call :make_junction "%bundle_source_dir%\runtimes\nodejs" "%repo_root%\runtimes\nodejs" || exit /b 1
exit /b 0

:check_rust_toolchain
set "cargo_probe="
for /f "usebackq delims=" %%V in (`"%~1" --version 2^>^&1`) do (
  if not defined cargo_probe set "cargo_probe=%%V"
)
if not defined cargo_probe set "cargo_probe=unknown cargo probe output"
"%~1" --version >nul 2>&1
if errorlevel 1 (
  echo [WARN] Cargo version probe failed: !cargo_probe!
  echo(!cargo_probe!| findstr /I /C:"rustup could not choose a version of cargo to run" /C:"no default toolchain configured" >nul
  if not errorlevel 1 (
    echo [FATAL] Cargo was found but no default Rust toolchain is configured.
    echo         Run:
    echo           rustup toolchain install stable-x86_64-pc-windows-msvc
    echo           rustup default stable-x86_64-pc-windows-msvc
    echo           rustup show active-toolchain
    exit /b 1
  )
  echo [FATAL] Cargo is installed but not runnable.
  echo         Details: !cargo_probe!
  exit /b 1
)

set "rustup_cmd="
set "active_toolchain="
if exist "%USERPROFILE%\.cargo\bin\rustup.exe" set "rustup_cmd=%USERPROFILE%\.cargo\bin\rustup.exe"
if not defined rustup_cmd (
  rustup --version >nul 2>&1
  if not errorlevel 1 set "rustup_cmd=rustup"
)
if not defined rustup_cmd (
  echo [WARN] rustup not found; skipping explicit default-toolchain validation.
  exit /b 0
)
for /f "delims=" %%V in ('"%rustup_cmd%" show active-toolchain 2^>nul') do set "active_toolchain=%%V"
if not defined active_toolchain (
  echo [FATAL] Cargo was found but no active Rust toolchain is configured.
  echo         Run:
  echo           rustup toolchain install stable-x86_64-pc-windows-msvc
  echo           rustup default stable-x86_64-pc-windows-msvc
  echo           rustup show active-toolchain
  exit /b 1
)
echo !active_toolchain! | findstr /I /C:"no default toolchain" >nul
if not errorlevel 1 (
  echo [FATAL] Cargo was found but no default Rust toolchain is configured.
  echo         Run:
  echo           rustup toolchain install stable-x86_64-pc-windows-msvc
  echo           rustup default stable-x86_64-pc-windows-msvc
  echo           rustup show active-toolchain
  exit /b 1
)
echo [INFO] Rust active toolchain: !active_toolchain!
exit /b 0

:make_junction
cmd /c mklink /J "%~1" "%~2" >nul
if errorlevel 1 (
  echo [FATAL] Failed to create junction "%~1" -> "%~2".
  exit /b 1
)
exit /b 0

:cleanup_bundle_sources
if exist "%bundle_source_dir%" rd /s /q "%bundle_source_dir%" >nul 2>&1
exit /b 0

:build_error
call :cleanup_bundle_sources
if /I "%CI%"=="1" endlocal & exit /b 1
if /I "%CI%"=="true" endlocal & exit /b 1
echo.
echo Press any key to close this build script...
pause >nul
endlocal & exit /b 1
