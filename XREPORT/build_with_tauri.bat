@echo off
setlocal enabledelayedexpansion

set "project_folder=%~dp0"
set "client_dir=%project_folder%client"
set "tauri_dir=%client_dir%\src-tauri"
set "bundle_dir=%tauri_dir%\target\release\bundle"
set "release_export_dir=%project_folder%..\release\windows"
set "runtime_node_dir=%project_folder%resources\runtimes\nodejs"

echo [TAURI] Release build helper

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
for /f "delims=" %%V in ('"%cargo_cmd%" --version 2^>nul') do set "cargo_version=%%V"
echo [INFO] Cargo command: %cargo_cmd%
if defined cargo_version echo [INFO] !cargo_version!
if /I not "%cargo_cmd%"=="cargo" (
  for %%I in ("%cargo_cmd%") do set "PATH=%%~dpI;%PATH%"
)
set "CARGO=%cargo_cmd%"

echo [CHECK] Resolving npm...
set "npm_cmd="
if exist "%runtime_node_dir%\npm.cmd" set "npm_cmd=%runtime_node_dir%\npm.cmd"
if not defined npm_cmd (
  npm --version >nul 2>&1
  if not errorlevel 1 set "npm_cmd=npm"
)
if not defined npm_cmd (
  echo [FATAL] npm not found. Install Node.js or run XREPORT\start_on_windows.bat once.
  goto build_error
)

echo [CHECK] Resolving node...
set "node_cmd="
if exist "%runtime_node_dir%\node.exe" set "node_cmd=%runtime_node_dir%\node.exe"
if not defined node_cmd (
  node --version >nul 2>&1
  if not errorlevel 1 set "node_cmd=node"
)
if not defined node_cmd (
  for %%I in ("%npm_cmd%") do (
    if exist "%%~dpInode.exe" set "node_cmd=%%~dpInode.exe"
  )
)
if not defined node_cmd (
  echo [FATAL] node not found. Install Node.js or run XREPORT\start_on_windows.bat once.
  goto build_error
)
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

echo [STEP 2/2] Building Tauri application
if exist "%release_export_dir%" (
  echo [INFO] Removing previous exported release folder: "%release_export_dir%"
)
echo [CMD] "%npm_cmd%" run tauri:build:release
call "%npm_cmd%" run tauri:build:release
if errorlevel 1 (
  popd >nul
  echo [FATAL] Tauri build failed.
  goto build_error
)
popd >nul

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

:build_error
echo.
echo Press any key to close this build script...
pause >nul
endlocal & exit /b 1

