(() => {
  const activateDesktopStartup = () => {
    if (window.location.protocol !== 'tauri:') return;
    const panel = document.getElementById('desktop-startup');
    const application = document.querySelector('app-root');
    if (panel) panel.hidden = false;
    if (application) application.hidden = true;
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', activateDesktopStartup, { once: true });
  } else {
    activateDesktopStartup();
  }
})();
