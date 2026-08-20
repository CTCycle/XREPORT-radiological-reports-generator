use std::ffi::c_void;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS};

#[cfg(windows)]
pub struct SingleInstance {
    handle: *mut c_void,
}

#[cfg(not(windows))]
pub struct SingleInstance;

#[cfg(windows)]
unsafe impl Send for SingleInstance {}

#[cfg(windows)]
unsafe impl Sync for SingleInstance {}

#[cfg(windows)]
impl SingleInstance {
    pub fn acquire() -> Result<Option<Self>, String> {
        let name: Vec<u16> = "Local\\io.github.ctcycle.xreport.desktop\0"
            .encode_utf16()
            .collect();
        let handle = unsafe {
            windows_sys::Win32::System::Threading::CreateMutexW(std::ptr::null(), 1, name.as_ptr())
        };
        if handle.is_null() {
            return Err("Unable to create the XREPORT desktop instance mutex".to_string());
        }
        if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            unsafe { windows_sys::Win32::Foundation::CloseHandle(handle) };
            show_already_running_message();
            return Ok(None);
        }
        Ok(Some(Self {
            handle: handle.cast(),
        }))
    }
}

#[cfg(not(windows))]
impl SingleInstance {
    pub fn acquire() -> Result<Option<Self>, String> {
        Ok(Some(Self))
    }
}

#[cfg(windows)]
impl Drop for SingleInstance {
    fn drop(&mut self) {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.handle.cast()) };
    }
}

#[cfg(windows)]
fn show_already_running_message() {
    focus_existing_window();
    let title: Vec<u16> = "XREPORT is already running\0".encode_utf16().collect();
    let message: Vec<u16> = "The existing XREPORT desktop window is being used. Close it before starting another CPU or CUDA instance.\0"
        .encode_utf16()
        .collect();
    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::MessageBoxW(
            std::ptr::null_mut(),
            message.as_ptr(),
            title.as_ptr(),
            windows_sys::Win32::UI::WindowsAndMessaging::MB_OK
                | windows_sys::Win32::UI::WindowsAndMessaging::MB_ICONINFORMATION,
        );
    }
}

#[cfg(windows)]
fn focus_existing_window() {
    for caption in [
        "XREPORT — Radiological Reports (CPU)",
        "XREPORT — Radiological Reports (CUDA)",
        "XREPORT — Radiological Reports",
    ] {
        let caption: Vec<u16> = format!("{caption}\0").encode_utf16().collect();
        let window = unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::FindWindowW(
                std::ptr::null(),
                caption.as_ptr(),
            )
        };
        if !window.is_null() {
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::ShowWindow(window, 9);
                windows_sys::Win32::UI::WindowsAndMessaging::SetForegroundWindow(window);
            }
            return;
        }
    }
}
