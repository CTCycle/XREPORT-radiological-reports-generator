(() => {
  const key = 'theme-preference';
  const isPreference = (value) => value === 'light' || value === 'dark' || value === 'system';
  let preference = 'system';

  try {
    const stored = window.localStorage.getItem(key);
    if (isPreference(stored)) preference = stored;
  } catch {
    // Continue with the system preference when storage is unavailable.
  }

  let resolved = preference;
  if (resolved === 'system') {
    try {
      resolved = typeof window.matchMedia === 'function'
        && window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
    } catch {
      resolved = 'light';
    }
  }

  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
  document.getElementById('theme-color-meta')?.setAttribute(
    'content',
    resolved === 'dark' ? '#101419' : '#e9eef6',
  );
})();
