const startupStartedAt = typeof performance === 'undefined' ? 0 : performance.now();

export function markStartupPhase(phase: string): void {
  if (typeof performance === 'undefined') return;

  const elapsed = Math.round(performance.now() - startupStartedAt);
  performance.mark(`xreport-startup-${phase}`);
  console.info(`[startup] phase=${phase} elapsed_ms=${elapsed}`);
}
