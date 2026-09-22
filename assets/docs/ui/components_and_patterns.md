# UI Components And Patterns

Last updated: 2026-09-22

## Reusable Patterns

- Buttons, cards, forms, modal shells, dashboards, and navigation all follow tokenized spacing, color, radius, and shadow rules.
- Use `--focus-ring` for visible keyboard focus feedback.
- Disabled controls must communicate both visually and behaviorally.

### Application startup gate

The root shell owns a dedicated startup gate before it creates the
`RouterOutlet`. `StartupReadinessService` polls `/api/health` serially and
keeps route components, including the inference catalogue request, out of the
DOM until the response contains `status: "ok"`. The `StartupScreenComponent`
provides the shared XREPORT radiograph-to-report loading surface, phase copy,
retry action, accessibility status, and ready exit transition. This is a
shell-level lifecycle component, not a feature-page loading state.

## Required Interactive States

- default
- hover
- active or selected where applicable
- disabled
- focus-visible

## Component Patterns In Active Use

### Navigation

- `app-nav-button`
- `sidebar-link`
- active route styling

### Forms

- checkbox and toggle variants
- disabled state handling

### Modals And Dialogs

- overlay or backdrop layer
- centered modal layout
- explicit close actions
- keyboard-focus-safe controls

### Contextual Guidance

- `GuidanceService` stores versioned, per-item guidance state under `xreport.guidance.v1` through `StorageService`.
- `app-feature-tip` is reserved for first-use callouts on genuinely non-obvious workflows; it is dismissible and does not repeat after its content version is seen.
- `app-help-popover` provides short, local explanations for model contracts, dataset processing, checkpoint actions, and validation configuration.
- Keep short contextual help in the relevant section title or field label; do not reserve a separate helper row for a lone information control.
- `app-guided-tour` consumes declarative `GuidanceDefinition` and `TourStep` data. Tours can be closed before completion, navigated backward and forward, and replayed from Help & tips.
- `app-tips-and-tricks` is the low-priority manual entry point for concise workflow reminders and the lightweight inference demonstration.
- Guidance anchors use `data-guidance-target` attributes and must remain attached to the relevant control or region when templates change.

### Data Views

- dashboard cards
- chart sections
- progress bars
- report history cards and detail editors
- report modals

## Route-Level Page Structure

- Dataset page: dataset loading, preprocessing, browsing, and validation entry actions
- Training page: training start or resume, checkpoint management, and metrics dashboards
- Inference page: filterable local model catalog and details, capability-aware study preparation, clinical context, generation profiles, and editable model-declared report sections with copy, regenerate, and export actions
- Reports page: filterable persisted inference sessions, explicit loading/empty/error states, paginated summaries, and links to detail views
- Report detail page: session metadata, reusable section-aware draft editor, original-output disclosure, atomic save, and destructive session deletion
- Dataset validation page: validation orchestration and report review
- Settings page: one-column runtime configuration form with General, Data
  access, and Advanced sections; save sends only changed fields and reset is
  delegated to the backend.

## Layout Composition

- `MainLayout` provides top branding, primary navigation, and routed content.
- Route pages own functional modules while reusing shared components for consistency.

`ReportDraftEditorComponent` is shared by the live inference draft and
persisted report detail page. It renders the model-declared output sections,
keeps raw-report output intact, and supports read-only rendering when a
session is not in a successful editable state.

The Settings route reuses the existing footer gear navigation control and
`app-nav-button` active styling. It does not duplicate the theme selector or
expose deployment, database, secret, or static inference policy values.
