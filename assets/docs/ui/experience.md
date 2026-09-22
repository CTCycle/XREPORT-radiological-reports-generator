# UI Experience Standards

Last updated: 2026-09-22

## Core UX Journeys

- Preserve dataset to training to inference to validation flow continuity.
- Keep long-running task behavior consistent:
  - start action
  - visible progress or loading
  - terminal success, error, or cancel feedback
- Keep the inference research-use warning visible above the drafting workspace. Model status and capabilities must be clear before image upload or generation.
- Generated report text remains an editable draft using the output sections declared by the selected model. Models may expose Findings, Impression, both sections, or a raw report; it is never presented as a clinically approved result.
- Keep completed inference sessions discoverable from Reports. The list exposes model, status, images, timing, and a short preview; the detail page exposes provenance and metadata while preserving generated output separately from user edits.

## Application startup

- The browser interface opens as soon as the frontend preview is reachable;
  backend initialization continues behind the shell-level startup surface.
- The startup surface uses four states: `starting`, `slow`, `unavailable`, and
  `ready`. It does not claim that a backend has crashed when the browser can
  only observe that readiness has not arrived.
- `slow` appears after approximately 15 seconds. `unavailable` appears after
  60 seconds, keeps automatic low-frequency polling active, and exposes
  **Retry connection**. A later successful `/api/health` response recovers
  without a browser refresh.
- Only a health response with `status: "ok"` unlocks routing. The startup gate
  is terminal for that application instance and never returns for ordinary
  feature-level API errors after the workspace is visible.
- The radiograph illustration and report bars are decorative. The live status
  text is exposed through an accessible status region, and reduced-motion users
  receive a static composition with only the short ready transition.
- The startup illustration uses a restrained structural radiograph and one
  scanning accent; decorative marker dots and enclosing card chrome are not
  required for readiness communication.

## Interaction Consistency

- Use consistent button labels and affordances for primary, cancel, and destructive actions.
- Keep modal close behavior predictable.
- Keep empty, loading, and error states explicit. Avoid silent failures.

## Reports history

- Reports is a durable review surface for persisted inference sessions; source
  radiographs are not retained there.
- List filters are explicit and reversible: model reference, lifecycle status,
  sort order, and pagination.
- A successful session can be edited by report image and declared output
  section. Save is atomic, disabled until a draft changes, and never replaces
  the generated model output. Deleting a session is explicit and returns the
  user to the history list.

## Theme selection

- The navigation theme selector exposes Light, Dark, and System preferences.
- Desktop navigation uses equal icon targets with native tooltips; mobile navigation keeps the text labels visible.
- The selected preference is shown with a selected state and `aria-pressed`; the control remains keyboard accessible on desktop and mobile.
- System follows the browser or operating-system `prefers-color-scheme` setting while selected. Manual preferences are not overridden by OS changes.
- Theme changes apply immediately without a page refresh and remain functional when the backend is unavailable.

## Runtime settings

- Settings is available from the existing navigation footer gear and remains
  keyboard accessible on desktop and mobile.
- The page uses explicit loading, load-error, dirty, validation, saving,
  resetting, success, and failure states.
- Save is disabled until a valid change exists and sends only changed fields.
- Reset asks the backend for authoritative defaults rather than calculating a
  second frontend copy of them.
- Help text explains that seed, polling interval, and inference timeout changes
  apply to new work; active jobs and generations retain their starting values.

## Contextual Guidance

- Keep onboarding optional and concentrated on the inference and dataset workflows where users must perform several non-obvious steps.
- Use short first-use callouts, local help popovers, and a small replayable inference tour instead of automatic tours on every route.
- Training keeps its existing five-step wizard; validation keeps its existing metric descriptions. Add only focused help where terminology or configuration choices may be unclear.
- Persist seen, dismissed, skipped, and completed states by content version. Manual replay remains available from the sidebar Help & tips action.
- Guidance overlays must support Escape and keyboard focus restoration, avoid obscuring the highlighted control, adapt to narrow screens, and remain static when reduced motion is requested.

## Responsiveness

- Ensure all primary flows remain usable on narrower widths.
- Common responsive adjustments include stacking multi-column sections, reducing card widths, and avoiding horizontal overflow in controls and tables where possible.
- Minimum supported layout assumptions stay around a `320px` body min-width baseline from current global CSS.

## Accessibility

- Keyboard navigation must remain functional for major workflows.
- Keep `:focus-visible` styles for buttons, inputs, links, selects, and textareas.
- Preserve existing ARIA usage in UI components such as modal dialogs, labeled icon buttons, and toggle state controls.
- Icon-only actions must include `aria-label`.
- Respect reduced-motion preference through existing `prefers-reduced-motion` handling.

## Design Principles

- Consistency first. Reuse existing tokens and components before introducing variants.
- Favor clarity over visual novelty in workflow-heavy pages.
- Keep feedback for long-running operations and validation outcomes predictable.
- Keep UI complexity proportional to clinical and technical task needs.
