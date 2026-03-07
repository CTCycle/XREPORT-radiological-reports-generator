@echo off
setlocal enabledelayedexpansion

set "project_folder=%~dp0"
set "client_dir=%project_folder%client"
set "tauri_dir=%client_dir%\src-tauri"
set "bundle_dir=%tauri_dir%\target\release\bundle"

echo [TAURI] Release build helper

where cargo >nul 2>&1
if errorlevel 1 (
  echo [FATAL] Rust/Cargo not found. Install Rust first: https://rustup.rs/
  goto build_error
)

set "npm_cmd="
where npm >nul 2>&1
if not errorlevel 1 set "npm_cmd=npm"
if not defined npm_cmd if exist "%project_folder%resources\runtimes\nodejs\npm.cmd" (
  set "npm_cmd=%project_folder%resources\runtimes\nodejs\npm.cmd"
)
if not defined npm_cmd (
  echo [FATAL] npm not found. Install Node.js or run XREPORT\start_on_windows.bat once.
  goto build_error
)

if not exist "%client_dir%\package.json" (
  echo [FATAL] Missing client package.json at "%client_dir%"
  goto build_error
)

pushd "%client_dir%" >nul
if exist "package-lock.json" (
  call "%npm_cmd%" ci
) else (
  call "%npm_cmd%" install
)
if errorlevel 1 (
  popd >nul
  echo [FATAL] npm dependency installation failed.
  goto build_error
)

call "%npm_cmd%" run tauri:build
if errorlevel 1 (
  popd >nul
  echo [FATAL] Tauri build failed.
  goto build_error
)
popd >nul

echo [OK] Build completed successfully.
if exist "%bundle_dir%" (
  echo [INFO] Release artifacts:
  echo        %bundle_dir%
) else (
  echo [WARN] Build finished but bundle directory was not found:
  echo        %bundle_dir%
)

endlocal & exit /b 0

:build_error
echo.
echo Press any key to close this build script...
pause >nul
endlocal & exit /b 1
