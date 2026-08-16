# TypeScript And Frontend Rules

Last updated: 2026-06-03

Apply these rules to Angular, TypeScript, and frontend integration work.

## Baseline

- Stack: standalone Angular 22 plus strict TypeScript
- Keep TypeScript strictness aligned with the existing `tsconfig` settings.
- Preserve the existing feature structure:
  - `src/app/pages`
  - `src/app/components`
  - `src/app/services`
  - `src/app/types`

## Typing And State

- Avoid `any` unless it is unavoidable and tightly scoped.
- Model API payloads and job contracts with explicit types or interfaces.
- Keep route-level state local when possible.
- Use shared context or hooks only when state genuinely spans pages.

## API Integration

- Route API calls through typed `src/app/services/*` modules.
- Keep `/api` contract alignment with backend endpoints and response models.
- Preserve the established start, poll, and cancel pattern for long-running UX.

## Styling And Components

- Reuse existing tokens and CSS variables from `src/index.css`.
- Keep component CSS colocated by feature or component where that pattern already exists.
- Preserve keyboard focus visibility plus disabled and loading states for interactive controls.
