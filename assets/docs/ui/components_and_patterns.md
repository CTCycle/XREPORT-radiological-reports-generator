# UI Components And Patterns

Last updated: 2026-10-27

## Reusable Patterns

- All eight routed screens render inside `MainLayout` and use `.page-container` for the shared 1400px page cap and responsive gutter. Page styles should define internal grids, not route-specific content origins.
- `MainLayout` keeps the 208px desktop navigation and a stable routed-content scroll area. Keep shell dimensions independent of the active route.
- `.btn`, `.primary-button`, `.secondary-button`, `.danger-button`, `.form-input`, and `.form-select` are global shared patterns; extend them with layout-specific classes instead of restating their base dimensions and colors.
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
- active route styling

### Status And Notifications

- `app-system-status-bar` is a full-width strip below the navigation and content
  (Backend and CUDA LEDs, RAM and VRAM in GB), fed by `SystemStatusService`.
- `NotificationService` keeps the notification history (local storage, capped at 50)
  and the transient toasts rendered by `app-toast-host`. Pass `history: false` for
  toast-only confirmations. Theme changes apply silently and raise no toast on
  success. `app-notification-center` in the
  sidebar footer lists the history with unread badge, mark-read, dismiss and clear.
- Inference completion, failure and cancellation are emitted by the inference page
  while it is mounted; connection loss and restoration by `SystemStatusService`.
- Report detail renders generation provenance through `app-generation-metadata`
  (grouped, labelled fields) with the stored JSON available under "Raw JSON".

### Forms

- checkbox and toggle variants
- disabled state handling

### Modals And Dialogs

- overlay or backdrop layer
- centered modal layout
- explicit close actions
- keyboard-focus-safe controls
- Studio metadata dialogs (dataset and checkpoint) open in a grouped "Human-readable" view by default, with a "JSON" toggle and a "Copy JSON" action that always copies the original metadata unchanged. Grouping and labels come from `common/metadata-view.ts`; unknown keys stay visible under "Other".

### Contextual Guidance

- `GuidanceService` stores versioned, per-item guidance state under `xreport.guidance.v1` through `StorageService`.
- `app-feature-tip` is reserved for first-use callouts on genuinely non-obvious workflows; it is dismissible and does not repeat after its content version is seen.
- `app-help-popover` provides short, local explanations for model contracts, dataset processing, checkpoint actions, and validation configuration.
- Keep short contextual help in the relevant section title or field label; do not reserve a separate helper row for a lone information control.
- `app-guided-tour` consumes declarative `GuidanceDefinition` and `TourStep` data. Tours can be closed before completion, navigated backward and forward, and replayed from Help & tips.
- `app-tips-and-tricks` is the low-priority manual entry point for concise workflow reminders and the lightweight inference demonstration. In Studio it hosts a compact five-step dataset-to-training list with status chips; a failed live-status call collapses to one short unavailable line.
- Guidance anchors use `data-guidance-target` attributes and must remain attached to the relevant control or region when templates change.

### Data Views

- dashboard cards
- chart sections
- progress bars
- report history cards and detail editors
- report modals
- Studio Training page lists are tables (header, fixed-height rows, in-table italic empty row, matching the Dataset page); action buttons never wrap their labels and wrap to a new line instead.
- Studio training dashboard states: idle, initializing, running, completed, failed, interrupted. They derive from the polled job status (`sessionStatus`) plus worker phase. Unavailable metrics render as `—`, never as zero; real zeros appear only after training data exists.
- Dashboard charts are inline SVG with train (solid) and validation (dashed) series, labelled axes, a hover/keyboard tooltip, and a width that follows the container. The log marks warnings and errors with a label as well as colour and keeps the latest entry in view unless scrolled up.

## Route-Level Page Structure

- Dataset page: dataset loading, preprocessing, browsing, and validation entry actions
- Model page (Studio): three zones — preset library, architecture canvas (statistics, dataset sizing, SVG diagram from the backend graph; two lanes when there is room, one stacked column otherwise), and a contextual inspector for the selected component with help, impact badges, an Advanced disclosure, inline validation and a compatibility card. Edits preview through a debounced, latest-wins request; invalid drafts dim the last valid diagram and cannot be saved.
- Training page: training start or resume, checkpoint management, and metrics dashboards. The wizard's Model step selects an architecture preset.
- Inference page: filterable local model catalog and details, capability-aware study preparation, clinical context, generation profiles (editable after a draft exists), described run scopes, a report selector for per-image batches, and editable model-declared report sections with copy, regenerate, and export actions
- Reports page: persisted inference sessions whose model-reference (debounced), status, and sort filters apply immediately, explicit loading/empty/error states, paginated summaries, and links to detail views
- Report detail page: session metadata, reusable section-aware draft editor, original-output disclosure that bolds and underlines words removed or changed in the current draft, atomic save, and destructive session deletion
- Dataset validation page: validation orchestration and report review
- Settings page: one-column runtime settings screen. Inference shows General, Advanced,
  Model access, Storage, and Database; Studio shows only General, Jobs, and Database
  (the storage root is shared with Inference, so it is not offered there). Non-database settings persist automatically
  with debounced numeric inputs; database settings remain drafts until the
  current candidate passes a connection test and the user confirms the switch.

## Layout Composition

- `MainLayout` owns product branding, left-side primary navigation, the routed-content scroll area, and shared help/theme controls.
- Route pages own functional modules inside the shared page container while reusing global controls and shared components for consistency.

`ReportDraftEditorComponent` is shared by the live inference draft and
persisted report detail page. It renders the model-declared output sections,
keeps raw-report output intact, and supports read-only rendering when a
session is not in a successful editable state.

The Inference page takes its save baseline from the canonical report-history
detail after a generation job completes. It sends edits through the existing
history update API, maps a shared study report to its persisted row, and guards
unsaved changes at the route and browser-exit boundaries. No separate draft
store is used.

The Settings route reuses the existing footer gear navigation control and
`app-nav-button` active styling. It does not duplicate the theme selector or
expose deployment, secret, or static inference policy values. Database
connection settings live in the dedicated Database section and are stored
outside the active database.
