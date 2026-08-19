# UI Design Tokens

Last updated: 2026-08-20

Use `app/client/src/styles.css` as the source of truth for UI tokens.

## Typography

- Base font stack:
  - `'Space Grotesk', 'Segoe UI', 'Inter', sans-serif`
- Root line height: `1.5`
- Core size tokens:
  - `--text-xs: 0.75rem`
  - `--text-sm: 0.875rem`
  - `--text-base: 1rem`

Readability rules:

- Use `--text-base` for primary content.
- Use `--text-sm` for controls and secondary labels.
- Use `--text-xs` only for metadata and helper text.
- Do not override the global font family per page without a product-level design change.

## Layout And Spacing

- Use tokenized spacing from `--space-1` through `--space-8`.
- Keep an 8px-aligned rhythm for margins, paddings, and grouped controls.
- Route pages use bounded content containers with `max-width` patterns in page CSS.
- Keep section and card compositions consistent through tokenized gaps and internal spacing.
- Interactive elements should align to established heights:
  - `--control-height-sm: 32px`
  - `--control-height-md: 36px`

## Breakpoints

- Existing breakpoint family includes `1200px`, `1024px`, `960px`, `768px`, `720px`, `640px`, and `480px`.
- New responsive behavior should align with this breakpoint family unless a broader refactor is planned.

## Color, Radius, And Shadow

Background and surface tokens:

- `--bg-primary`
- `--bg-secondary`
- `--bg-muted`
- `--surface-elevated`
- `--hover-bg`
- `--selected-bg`

Text tokens:

- `--text-primary`
- `--text-secondary`
- `--text-muted`

Border and input tokens:

- `--border-color`
- `--border-subtle`
- `--input-bg`
- `--input-border`

Accent and semantic tokens:

- `--accent-color`
- `--accent-color-strong`
- `--accent-fill`
- `--link-color`
- `--link-hover`
- `--success-color`
- `--warning-color`
- `--error-color`
- `--success-bg`
- `--warning-bg`
- `--error-bg`
- `--disabled-bg`
- `--disabled-text`
- `--overlay-color`
- `--code-bg`
- `--media-bg`
- `--on-accent`
- `--on-media`
- `--chart-primary`
- `--chart-secondary`

Shape and elevation tokens:

- `--radius-sm`
- `--radius-md`
- `--radius-lg`
- `--shadow-sm`
- `--shadow-md`
- `--focus-ring`

Accessibility rule:

- Avoid one-off colors when an existing semantic or tokenized value already exists.

## Theme behavior

- The resolved theme is applied to the root HTML element as `data-theme="light"` or `data-theme="dark"`.
- Theme preferences are `light`, `dark`, or `system`; the default preference is `system`.
- The selected preference is stored locally under `theme-preference`. The resolved theme is never stored as the preference, so `system` continues following the OS.
- `color-scheme`, native controls, the browser theme color, overlays, status states, charts, and form surfaces must consume the active token palette.
- Component styles should use semantic variables instead of duplicating light or dark color literals.
- Intentional media-viewer stages may use `--media-bg`; all surrounding controls and overlays must remain tokenized.
