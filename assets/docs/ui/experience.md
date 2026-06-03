# UI Experience Standards

Last updated: 2026-06-03

## Core UX Journeys

- Preserve dataset to training to inference to validation flow continuity.
- Keep long-running task behavior consistent:
  - start action
  - visible progress or loading
  - terminal success, error, or cancel feedback

## Interaction Consistency

- Use consistent button labels and affordances for primary, cancel, and destructive actions.
- Keep modal close behavior predictable.
- Keep empty, loading, and error states explicit. Avoid silent failures.

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
