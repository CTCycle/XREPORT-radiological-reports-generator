use tauri::Url;

pub fn is_allowed_navigation(url: &Url, backend_port: Option<u16>) -> bool {
    if url.scheme() == "tauri" {
        return true;
    }
    if url.scheme() == "http"
        && url.host_str() == Some("127.0.0.1")
        && backend_port.is_some_and(|port| url.port() == Some(port))
    {
        return true;
    }
    if url.scheme() == "https"
        && matches!(
            url.host_str(),
            Some("huggingface.co") | Some("www.huggingface.co")
        )
        && (url.path().starts_with("/models/")
            || url.path().starts_with("/datasets/")
            || url.path().starts_with("/spaces/"))
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::is_allowed_navigation;

    #[test]
    fn only_expected_backend_and_hugging_face_urls_are_allowed() {
        assert!(is_allowed_navigation(
            &tauri::Url::parse("http://127.0.0.1:5003/").unwrap(),
            Some(5003)
        ));
        assert!(is_allowed_navigation(
            &tauri::Url::parse("https://huggingface.co/models/example").unwrap(),
            Some(5003)
        ));
        assert!(!is_allowed_navigation(
            &tauri::Url::parse("http://127.0.0.1:5004/").unwrap(),
            Some(5003)
        ));
        assert!(!is_allowed_navigation(
            &tauri::Url::parse("file:///C:/Windows/System32/cmd.exe").unwrap(),
            Some(5003)
        ));
    }
}
