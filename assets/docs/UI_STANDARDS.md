# UI Standards

Last updated: 2026-04-24

## 1. Typography

- Base font stack from `src/index.css`:
  - `'Space Grotesk', 'Segoe UI', 'Inter', sans-serif`
- Root line height: `1.5`
- Core size tokens:
  - `--text-xs: 0.75rem`
  - `--text-sm: 0.875rem`
  - `--text-base: 1rem`
- Readability rules:
  - Use body-sized text (`--text-base`) for primary content.
  - Use `--text-sm` for controls/secondary labels.
  - Use `--text-xs` only for metadata/helper text.
  - Do not override global font family per page without a product-level design change.

## 2. Layout and Spacing

### Spacing scale

- Use tokenized spacing:
  - `--space-1` to `--space-8` (`4px` to `32px` equivalent)
- Keep an 8px-aligned rhythm for margins/paddings and control grouping.

### Grid and container behavior

- Route pages use bounded content containers (`max-width` patterns in page CSS).
- Keep section/card compositions with consistent internal spacing and tokenized gaps.
- Keep interactive elements aligned to consistent control heights:
  - `--control-height-sm: 32px`
  - `--control-height-md: 36px`

### Breakpoints in current UI

- Existing responsive breakpoints include: `1200px`, `1024px`, `960px`, `768px`, `720px`, `640px`, `480px`.
- New responsive behavior should align to this breakpoint family unless a broader responsive refactor is planned.

## 3. Color System

Use root variables from `src/index.css` as the source of truth.

### Primary and neutral

- Backgrounds/surfaces:
  - `--bg-primary`, `--bg-secondary`, `--bg-muted`
- Text:
  - `--text-primary`, `--text-secondary`
- Borders:
  - `--border-color`
- Inputs:
  - `--input-bg`

### Accent and semantic

- Primary accent:
  - `--accent-color`
  - `--accent-color-strong`
- Semantic status:
  - `--success-color`
  - `--warning-color`
  - `--error-color`

### Contrast and accessibility

- Ensure text/background combinations meet accessibility contrast expectations.
- Avoid introducing one-off colors when a semantic/tokenized value already exists.

## 4. Components and Interaction Patterns

## 4.1 Reusable patterns

- Buttons, cards, forms, modal shells, dashboards, and navigation all follow tokenized spacing/color/radius.
- Radius tokens:
  - `--radius-sm`, `--radius-md`, `--radius-lg`
- Shadow tokens:
  - `--shadow-sm`, `--shadow-md`

## 4.2 States

- Required interactive states:
  - default
  - hover
  - active/selected (where applicable)
  - disabled
  - focus-visible
- Focus visibility:
  - use `--focus-ring` and never remove visible keyboard focus feedback.
- Disabled controls:
  - must communicate both visually and behaviorally (non-clickable).

## 4.3 Specific component classes in use

- Navigation:
  - `app-nav-button`, `sidebar-link`, active route styling
- Form patterns:
  - checkbox/toggle variants and disabled states
- Modal/dialog patterns:
  - overlay/backdrop + centered modal + close actions + keyboard-focus-safe controls
- Data views:
  - dashboard cards, chart sections, progress bars, report modals

## 5. Page Structure

Primary page roles:

- Dataset page (`/dataset`):
  - dataset loading, preprocessing, browsing, validation entry actions
- Training page (`/training`):
  - training start/resume, checkpoint management, progress/metrics dashboards
- Inference page (`/inference`):
  - image selection/upload, checkpoint and generation-mode selection, report outputs
- Dataset validation page (`/dataset/validate/:datasetName`):
  - validation workflow orchestration and report review

Layout composition:

- App shell in `MainLayout` contains top branding + primary nav + routed content.
- Each route page owns its functional modules while reusing shared components for consistency.

## 6. User Experience Standards

Core UX journeys to preserve:

- Dataset -> Training -> Inference -> Validation flow continuity
- Consistent long-running task behavior:
  - start action
  - visible progress/loading
  - terminal success/error/cancel feedback

Interaction consistency rules:

- Use consistent button labels and affordances for destructive, cancel, and primary actions.
- Keep modal close behavior predictable (close button + backdrop behavior where already implemented).
- Keep empty/loading/error states explicit; avoid silent failures.

## 7. Responsiveness

- Ensure all primary flows remain usable on narrower widths.
- Common responsive adjustments:
  - stack multi-column sections
  - reduce card widths
  - avoid horizontal overflow in controls/tables where possible
- Minimum supported layout assumptions:
  - body min width baseline around `320px` (from current global CSS)

## 8. Accessibility

- Keyboard navigation must remain functional for major workflows.
- Keep `:focus-visible` styles for buttons/inputs/links/selects/textarea.
- Preserve ARIA usage already present in UI components (for modal dialogs, labeled icon buttons, toggle state).
- Icon-only actions must include `aria-label`.
- Respect reduced-motion preference (`prefers-reduced-motion` rules are already present and must be kept).

## 9. Design Principles

- Consistency first: reuse existing tokens/components before introducing variants.
- Clarity over visual novelty in workflow-heavy pages.
- Predictability for long-running operations and validation/error feedback.
- Keep UI complexity proportional to clinical/technical task needs; avoid decorative elements that reduce readability or operational speed.
