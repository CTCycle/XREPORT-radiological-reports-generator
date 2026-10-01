(() => {
  const query = new URLSearchParams(window.location.search);
  const logPath = query.get('log');
  const status = document.getElementById('status');
  const log = document.getElementById('log-path');
  // Native startup details stay in desktop-shell.log; keep the page copy human-readable.
  if (status) status.textContent = 'The local backend service did not become ready. Close and relaunch XREPORT to retry.';
  if (log && logPath) log.textContent = `Details were written to: ${logPath}`;
})();
