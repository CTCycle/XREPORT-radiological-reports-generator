(() => {
  const query = new URLSearchParams(window.location.search);
  const message = query.get('message');
  const logPath = query.get('log');
  const status = document.getElementById('status');
  const log = document.getElementById('log-path');
  if (status && message) status.textContent = message;
  if (log && logPath) log.textContent = `Details were written to: ${logPath}`;
})();
