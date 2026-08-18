# UI Components And Patterns

Last updated: 2026-08-18

## Reusable Patterns

- Buttons, cards, forms, modal shells, dashboards, and navigation all follow tokenized spacing, color, radius, and shadow rules.
- Use `--focus-ring` for visible keyboard focus feedback.
- Disabled controls must communicate both visually and behaviorally.

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
- `app-guided-tour` consumes declarative `GuidanceDefinition` and `TourStep` data. Tours can be skipped, closed, navigated backward, and replayed from Help & tips.
- `app-tips-and-tricks` is the low-priority manual entry point for concise workflow reminders and the lightweight inference demonstration.
- Guidance anchors use `data-guidance-target` attributes and must remain attached to the relevant control or region when templates change.

### Data Views

- dashboard cards
- chart sections
- progress bars
- report modals

## Route-Level Page Structure

- Dataset page: dataset loading, preprocessing, browsing, and validation entry actions
- Training page: training start or resume, checkpoint management, and metrics dashboards
- Inference page: filterable local model catalog and details, capability-aware study preparation, clinical context, generation profiles, and editable Findings/Impression drafting with copy, regenerate, and export actions
- Dataset validation page: validation orchestration and report review

## Layout Composition

- `MainLayout` provides top branding, primary navigation, and routed content.
- Route pages own functional modules while reusing shared components for consistency.
