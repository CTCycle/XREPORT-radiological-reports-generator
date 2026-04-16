# TypeScript and Frontend Guidelines (XREPORT)

Last updated: 2026-04-08

Repository-scoped standards for `XREPORT/client`.

## 1. Stack and Tooling Baseline

- React 18 + TypeScript + Vite 5
- Compiler strict mode is enabled (`XREPORT/client/tsconfig.json`)
- Build command also performs TypeScript checks:
  - `npm run build`

The repository currently keeps frontend tooling lightweight. Follow existing project setup instead of introducing new mandatory tooling unless requested.

## 2. Frontend Structure

Follow current module layout:
- `src/pages`: route pages
- `src/components`: reusable UI components
- `src/services`: backend API clients
- `src/hooks`: reusable hooks
- `src/AppStateContext.tsx`: shared application state

Preserve this structure when adding features.

## 3. API Integration Rules

- Use service modules in `src/services` for backend calls.
- Use `/api` paths consistently (frontend service modules and Vite proxy).
- Keep endpoint contracts aligned with FastAPI models and job polling behavior.

## 4. State and Job Flows

- Reuse `AppStateContext` for cross-page state.
- Reuse `usePersistedRecord` for persisted UI/job references.
- Long operations should use the established start + poll + terminal-status pattern.

## 5. Styling and Component Practices

- Reuse existing CSS file patterns (`*.css`) used by pages/components.
- Keep components focused and avoid large multipurpose files.
- Keep naming consistent with existing page/service/component conventions.

## 6. Type Safety Expectations

- Avoid `any` unless there is no practical typed alternative.
- Prefer explicit interfaces/types for request and response payloads.
- Narrow unknown runtime values before usage.

## 7. Validation Checklist for Frontend Changes

Before finalizing frontend edits:
1. Run `npm run build` in `XREPORT/client`.
2. Verify route-level behavior for impacted pages.
3. Verify API calls still match backend route paths and payload shapes.
