# UI Design Tokens

Last updated: 2026-06-03

Use `src/index.css` as the source of truth for UI tokens.

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

Text tokens:

- `--text-primary`
- `--text-secondary`

Border and input tokens:

- `--border-color`
- `--input-bg`

Accent and semantic tokens:

- `--accent-color`
- `--accent-color-strong`
- `--success-color`
- `--warning-color`
- `--error-color`

Shape and elevation tokens:

- `--radius-sm`
- `--radius-md`
- `--radius-lg`
- `--shadow-sm`
- `--shadow-md`

Accessibility rule:

- Avoid one-off colors when an existing semantic or tokenized value already exists.
