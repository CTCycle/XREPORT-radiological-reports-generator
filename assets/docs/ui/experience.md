# UI Experience Standards

Last updated: 2026-08-18

## Core UX Journeys

- Preserve dataset to training to inference to validation flow continuity.
- Keep long-running task behavior consistent:
  - start action
  - visible progress or loading
  - terminal success, error, or cancel feedback
- Keep the inference research-use warning visible above the drafting workspace. Model status and capabilities must be clear before image upload or generation.
- Generated report text remains an editable draft split into Findings and Impression; it is never presented as a clinically approved result.

## Interaction Consistency

- Use consistent button labels and affordances for primary, cancel, and destructive actions.
- Keep modal close behavior predictable.
- Keep empty, loading, and error states explicit. Avoid silent failures.

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
