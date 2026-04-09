# UI Standards (Frontend)

Last updated: 2026-04-08

## Spacing scale
- Base rhythm: 8px.
- Token set:
- `--space-1` = 4px
- `--space-2` = 8px
- `--space-3` = 12px
- `--space-4` = 16px
- `--space-5` = 20px
- `--space-6` = 24px
- `--space-8` = 32px

## Typography scale
- Body: `--text-base` (`1rem`).
- Small body / controls: `--text-sm` (`0.875rem`).
- Labels / metadata: `--text-xs` (`0.75rem`).
- Page containers inherit root font stack from `:root`; avoid page-level font overrides.

## Color system
- Surfaces and text: `--bg-primary`, `--bg-secondary`, `--input-bg`, `--text-primary`, `--text-secondary`.
- Borders: `--border-color`.
- Interactive accents: `--accent-color`, `--accent-color-strong`.
- Semantics: `--success-color`, `--warning-color`, `--error-color`.
- Focus: `--focus-ring` for keyboard-visible states.

## Component usage rules
- Buttons:
- Use shared `.btn` sizing and tokenized spacing.
- Ensure `:focus-visible` has visible ring and contrast.
- Icon-only controls:
- Minimum target is 32x32.
- Always provide `aria-label` and visible focus state.
- Inputs/selects:
- Use tokenized border/surface colors and consistent text sizing.
- Keep focus indication with border + ring, not color-only.
- Cards/panels:
- Prefer tokenized radii and subtle shadows; avoid introducing one-off shadow stacks.

## Do and Don't
- Do use tokens before introducing raw hex/rgba values.
- Do align spacing to the 8px rhythm.
- Do include reduced-motion fallback for repeated animations.
- Don't redefine global font family at page scope.
- Don't add new hardcoded brand/semantic colors when tokens exist.
- Don't ship interactive controls without keyboard-visible focus feedback.
