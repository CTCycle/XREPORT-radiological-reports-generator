# XREPORT Python 3.14.7 and inference layout validation

Date: 2026-09-21

## Runtime and launcher

- `start_on_windows.ps1 -Action Install` downloaded and hash-verified the
  Python 3.14.7 Windows embeddable runtime.
- The launcher detected the prior project venv was Python 3.14.2, removed only
  that disposable venv, and recreated it through `uv sync --frozen --all-extras`.
- The portable runtime and `app/server/.venv` both report Python 3.14.7.
- The locked environment synchronized 113 packages successfully.
- Frontend dependency installation and `ng build` completed successfully.
- Database initialization completed at Alembic head `f48a7c2e91b6`.

## Regression checks

- Baseline focused tests before the venv replacement: `9 passed`.
- Focused tests after the replacement: `9 passed`.
- Full backend unit suite after the replacement: `132 passed in 29.58s`.
- Launcher `-Action Launch` reported the environments ready, started backend and
  frontend services, and the live backend and inference route both returned
  HTTP 200.

## Inference section 1 layout

Rendered browser verification at `http://127.0.0.1:8003/inference` showed the
model catalogue constrained to the available viewport height with an inner
vertical scrollbar. Scrolling the left catalogue revealed the Custom XReport
checkpoints while the right model-details column remained in place. The
stacked/mobile breakpoint remains unconstrained by the desktop max-height.
