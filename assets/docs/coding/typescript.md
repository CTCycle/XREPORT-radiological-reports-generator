# TypeScript And Frontend Rules

Last updated: 2026-06-03

Apply these rules to React, TypeScript, and frontend integration work.

## Baseline

- Stack: React 18 plus TypeScript plus Vite
- Keep TypeScript strictness aligned with the existing `tsconfig` settings.
- Preserve the existing feature structure:
  - `src/pages`
  - `src/components`
  - `src/services`
  - `src/hooks`
  - `src/types`

## Typing And State

- Avoid `any` unless it is unavoidable and tightly scoped.
- Model API payloads and job contracts with explicit types or interfaces.
- Keep route-level state local when possible.
- Use shared context or hooks only when state genuinely spans pages.

## API Integration

- Route API calls through `src/services/*` modules.
- Keep `/api` contract alignment with backend endpoints and response models.
- Preserve the established start, poll, and cancel pattern for long-running UX.

## Styling And Components

- Reuse existing tokens and CSS variables from `src/index.css`.
- Keep component CSS colocated by feature or component where that pattern already exists.
- Preserve keyboard focus visibility plus disabled and loading states for interactive controls.
