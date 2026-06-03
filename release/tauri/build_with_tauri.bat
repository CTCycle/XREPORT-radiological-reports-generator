@echo off
setlocal enabledelayedexpansion

set "script_dir=%~dp0"
for %%I in ("%script_dir%..\..") do set "repo_root=%%~fI"
set "app_dir=%repo_root%\app"
set "client_dir=%app_dir%\client"
set "tauri_dir=%client_dir%\src-tauri"
set "bundle_dir=%tauri_dir%\target\release\bundle"
set "release_export_dir=%repo_root%\release\windows"
set "bundle_source_dir=%tauri_dir%\r"

set "runtime_python_exe=%repo_root%\runtimes\python\python.exe"
set "runtime_uv_exe=%repo_root%\runtimes\uv\uv.exe"
set "runtime_uv_lock=%repo_root%\app\server\uv.lock"
set "runtime_node_dir=%repo_root%\runtimes\nodejs"
set "node_cmd=%runtime_node_dir%\node.exe"
set "npm_cmd=%runtime_node_dir%\npm.cmd"
set "runtime_database_file=%app_dir%\resources\database.db"

echo [TAURI] Release build helper

echo [CHECK] Validating bundled runtimes...
if not exist "%runtime_python_exe%" (
  echo [FATAL] Missing embedded Python runtime at "%runtime_python_exe%"
  echo         Run start_on_windows.bat first to install portable runtimes.
  goto fail
)
echo [OK] embedded Python runtime found: %runtime_python_exe%

if not exist "%runtime_uv_exe%" (
  echo [FATAL] Missing embedded uv runtime at "%runtime_uv_exe%"
  echo         Run start_on_windows.bat first to install portable runtimes.
  goto fail
)
echo [OK] embedded uv runtime found: %runtime_uv_exe%

if not exist "%runtime_uv_lock%" (
  echo [FATAL] Missing backend uv lockfile at "%runtime_uv_lock%"
  echo         Run start_on_windows.bat first to install portable runtimes.
  goto fail
)
echo [OK] backend uv lockfile found: %runtime_uv_lock%

if not exist "%node_cmd%" (
  echo [FATAL] Missing embedded Node.js runtime at "%node_cmd%"
  echo         Run start_on_windows.bat first to install portable runtimes.
  goto fail
)
echo [OK] embedded Node.js runtime found: %node_cmd%

if not exist "%npm_cmd%" (
  echo [FATAL] Missing embedded npm runtime at "%npm_cmd%"
  echo         Run start_on_windows.bat first to install portable runtimes.
  goto fail
)
echo [OK] embedded npm runtime found: %npm_cmd%

if not exist "%app_dir%\server\pyproject.toml" (
  echo [FATAL] Missing backend pyproject.toml at "%app_dir%\server\pyproject.toml"
  echo         Run start_on_windows.bat first to install portable runtimes.
  goto fail
)
echo [OK] backend pyproject.toml found: %app_dir%\server\pyproject.toml

if not exist "%runtime_database_file%" (
  echo [FATAL] Missing runtime sqlite database at "%runtime_database_file%"
  echo         Run start_on_windows.bat first to install portable runtimes.
  goto fail
)
echo [OK] runtime sqlite database found: %runtime_database_file%

echo [CHECK] Resolving Cargo...
set "cargo_cmd="
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "cargo_cmd=%USERPROFILE%\.cargo\bin\cargo.exe"
if not defined cargo_cmd (
  cargo --version >nul 2>&1
  if not errorlevel 1 set "cargo_cmd=cargo"
)
if not defined cargo_cmd (
  echo [FATAL] Rust/Cargo not found. Install Rust first: https://rustup.rs/
  goto fail
)

"%cargo_cmd%" --version >nul 2>&1
if errorlevel 1 (
  echo [FATAL] Cargo is installed but not runnable.
  goto fail
)

set "rustup_cmd="
set "active_toolchain="
if exist "%USERPROFILE%\.cargo\bin\rustup.exe" set "rustup_cmd=%USERPROFILE%\.cargo\bin\rustup.exe"
if not defined rustup_cmd (
  rustup --version >nul 2>&1
  if not errorlevel 1 set "rustup_cmd=rustup"
)
if defined rustup_cmd (
  for /f "delims=" %%V in ('"%rustup_cmd%" show active-toolchain 2^>nul') do set "active_toolchain=%%V"
  if not defined active_toolchain (
    echo [FATAL] Cargo was found but no active Rust toolchain is configured.
    echo         Run:
    echo           rustup toolchain install stable-x86_64-pc-windows-msvc
    echo           rustup default stable-x86_64-pc-windows-msvc
    echo           rustup show active-toolchain
    goto fail
  )
  echo !active_toolchain! | findstr /I /C:"no default toolchain" >nul
  if not errorlevel 1 (
    echo [FATAL] Cargo was found but no default Rust toolchain is configured.
    echo         Run:
    echo           rustup toolchain install stable-x86_64-pc-windows-msvc
    echo           rustup default stable-x86_64-pc-windows-msvc
    echo           rustup show active-toolchain
    goto fail
  )
  echo [INFO] Rust active toolchain: !active_toolchain!
) else (
  echo [WARN] rustup not found; skipping explicit default-toolchain validation.
)

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
  goto fail
)

set "RUST_BACKTRACE=1"
set "CARGO_TERM_PROGRESS_WHEN=auto"

echo [CHECK] Cleaning stale Tauri release artifacts...
if exist "%bundle_source_dir%" rd /s /q "%bundle_source_dir%" >nul 2>&1
if exist "%tauri_dir%\target\release" (
  rd /s /q "%tauri_dir%\target\release" >nul 2>&1
  if exist "%tauri_dir%\target\release" (
    echo [FATAL] Failed to remove stale Tauri release directory "%tauri_dir%\target\release".
    goto fail
  )
)
if exist "%release_export_dir%" (
  rd /s /q "%release_export_dir%" >nul 2>&1
  if exist "%release_export_dir%" (
    echo [FATAL] Failed to remove stale Windows export directory "%release_export_dir%".
    goto fail
  )
)

echo [CHECK] Removing Python cache files from bundled source directories...
powershell -NoProfile -Command "$targets=@('%app_dir%\server\api','%app_dir%\server\common','%app_dir%\server\configurations','%app_dir%\server\domain','%app_dir%\server\learning','%app_dir%\server\repositories','%app_dir%\server\services','%app_dir%\scripts'); foreach($target in $targets){ Get-ChildItem -LiteralPath $target -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; Get-ChildItem -LiteralPath $target -Recurse -File -Filter '*.pyc' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue }"
if errorlevel 1 (
  echo [FATAL] Failed to remove Python cache files from bundled source directories.
  goto fail
)

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
  goto fail
)

echo [STEP 2/2] Building Tauri application
echo [CMD] "%npm_cmd%" run tauri:build:release
call "%npm_cmd%" run tauri:build:release
if errorlevel 1 (
  popd >nul
  echo [FATAL] Tauri build failed.
  goto fail
)
popd >nul

if exist "%bundle_source_dir%" rd /s /q "%bundle_source_dir%" >nul 2>&1

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

:fail
if exist "%bundle_source_dir%" rd /s /q "%bundle_source_dir%" >nul 2>&1
if /I "%CI%"=="1" endlocal & exit /b 1
if /I "%CI%"=="true" endlocal & exit /b 1
echo.
echo Press any key to close this build script...
pause >nul
endlocal & exit /b 1
