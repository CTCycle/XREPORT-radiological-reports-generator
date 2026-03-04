## WEB SEARCH
Use web search to verify facts and stay current on tools, frameworks, and industry standards when it improves accuracy.

## REQUIRED DOCUMENTATION REVIEW
Before any task, review the relevant files in `.agent/rules`:

- `GENERAL_RULES.md`, mandatory for every task
- `GUIDELINES_PYTHON.md`, when using Python
- `GUIDELINES_TYPESCRIPT.md`, when using TypeScript
- `GUIDELINES_TESTS.md`, when writing tests
- `ARCHITECTURE.md`, system structure and APIs
- `BACKGROUND_JOBS.md`, background job management
- `README_WRITING.md`, required README structure and standards

## SKILLS REFERENCE
When task-specific reusable workflows or capabilities are needed, check `.agent/skills/` and use the relevant skill guidance.

## DOCUMENTATION UPDATES
If changes materially affect behavior, architecture, or usage, update the relevant `.agent/rules` files and notify the user.

## CROSS-LANGUAGE PRINCIPLES

### Code quality
- Prefer consistent style, clear naming, and small single-purpose components.
- Optimize for readability, testability, and low coupling.

### Testing and automation
- Enforce CI checks: formatting, linting, type checks, tests, and security scans.

### Security
- Apply standard secure coding practices: input validation, correct auth handling, secret protection, minimal attack surface.

## EXECUTION RULES
- Use PowerShell by default for terminal commands in this repository.
- Use `cmd /c` only when invoking `.bat` scripts or CMD-specific syntax.

