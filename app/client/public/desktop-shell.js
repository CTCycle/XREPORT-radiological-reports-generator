(() => {
  const isDesktopAssetOrigin = () => window.location.protocol === 'tauri:'
    || (window.location.protocol === 'http:' && window.location.hostname === 'tauri.localhost');

  const activateDesktopStartup = () => {
    if (!isDesktopAssetOrigin()) return;
    const panel = document.getElementById('desktop-startup');
    const application = document.querySelector('app-root');
    if (panel) {
      panel.hidden = false;
      panel.classList.add('startup-shell-visible');
    }
    if (application) {
      application.hidden = true;
      application.setAttribute('aria-hidden', 'true');
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', activateDesktopStartup, { once: true });
  } else {
    activateDesktopStartup();
  }
})();
