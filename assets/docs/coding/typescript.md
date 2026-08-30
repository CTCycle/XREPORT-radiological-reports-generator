# TypeScript And Frontend Rules

Last updated: 2026-08-30

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
- Use shared services only when state genuinely spans pages. Server job state
  stays in the backend `JobManager`; pages keep transient UI state in memory.

## API Integration

- Route API calls through typed `src/app/services/*` modules.
- Keep `/api` contract alignment with backend endpoints and response models.
- Generate raw request/response types from the checked-in OpenAPI snapshot with
  `npm run generate:api`; do not hand-edit `src/app/types/api.generated.ts`.
- Use feature clients for feature endpoints and `JobsApiService` for the
  generic `/api/jobs` status and cancellation resource.
- Preserve the established start, poll, and cancel pattern for long-running UX.

## Styling And Components

- Reuse existing tokens and CSS variables from `src/index.css`.
- Keep component CSS colocated by feature or component where that pattern already exists.
- Preserve keyboard focus visibility plus disabled and loading states for interactive controls.
