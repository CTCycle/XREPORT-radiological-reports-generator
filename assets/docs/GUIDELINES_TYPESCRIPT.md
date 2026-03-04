## TypeScript (modern, 2024+ era) guidelines

### Compiler configuration and type safety
- Enable `strict` in `tsconfig.json` as the default posture for new codebases, then selectively relax only when justified. (TSConfig `strict`: https://www.typescriptlang.org/tsconfig/strict.html) (TSConfig reference: https://www.typescriptlang.org/tsconfig/)
- Prefer `unknown` over `any` when receiving untrusted or loosely typed inputs, then narrow with explicit checks. (TypeScript Narrowing: https://www.typescriptlang.org/docs/handbook/2/narrowing.html)

### Type system best practices
- Use control-flow based narrowing and type guards to model runtime checks explicitly and keep code safe. (TypeScript Narrowing: https://www.typescriptlang.org/docs/handbook/2/narrowing.html)
- Use generics to write reusable, type-safe components, and apply constraints only when required to preserve inference and usability. (TypeScript Generics: https://www.typescriptlang.org/docs/handbook/2/generics.html)

### Tooling baseline (common, widely adopted)
- Linting: ESLint with typescript-eslint for TypeScript syntax and type-aware rules. (typescript-eslint getting started: https://typescript-eslint.io/getting-started/) (typescript-eslint rules: https://typescript-eslint.io/rules/) (ESLint rules reference: https://eslint.org/docs/latest/rules/)
- Formatting: Prettier as an opinionated formatter to reduce style noise. (Prettier docs: https://prettier.io/docs/)
- Testing: Jest for unit tests across JS and TS ecosystems, with TypeScript integration as needed. (Jest getting started: https://jestjs.io/docs/getting-started)

### Testing practices
- Write tests that validate behavior, not implementation details, and keep unit tests fast enough to run frequently. (Jest docs: https://jestjs.io/docs/getting-started)

### Security
- Apply OWASP secure coding practices, validate and sanitize external inputs, and treat client-side code as untrusted from an attacker perspective. (OWASP Secure Coding Practices Quick Reference Guide: https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)

---

## Practical baseline checklists (copy-ready)

### Python baseline (recommended defaults)
- Style: PEP 8. (https://peps.python.org/pep-0008/)
- Formatter: Black, or Ruff formatter. (https://black.readthedocs.io/en/stable/the_black_code_style/) (https://docs.astral.sh/ruff/formatter/)
- Linter: Ruff. (https://docs.astral.sh/ruff/)
- Types: mypy for static checks. (https://mypy.readthedocs.io/)
- Tests: pytest. (https://docs.pytest.org/en/stable/getting-started.html)

### FastAPI baseline (recommended defaults)
- Dependencies: use `Depends` for auth, DB sessions, request-scoped resources. (https://fastapi.tiangolo.com/tutorial/dependencies/)
- Background work: use `BackgroundTasks` for post-response tasks, use workers for heavy jobs. (https://fastapi.tiangolo.com/tutorial/background-tasks/)
- Testing: use dependency overrides for isolation. (https://fastapi.tiangolo.com/advanced/testing-dependencies/)

### Java baseline (recommended defaults)
- Style guide: Google Java Style. (https://google.github.io/styleguide/javaguide.html)
- Build: Maven or Gradle, enforce in CI. (https://maven.apache.org/pom.html) (https://docs.gradle.org/current/userguide/java_plugin.html)
- Static analysis: Checkstyle, SpotBugs, PMD. (https://checkstyle.sourceforge.io/) (https://spotbugs.readthedocs.io/) (https://pmd.github.io/pmd/index.html)
- Testing: JUnit 5. (https://docs.junit.org/5.10.2/user-guide/)

### TypeScript baseline (recommended defaults)
- tsconfig: `strict: true`. (https://www.typescriptlang.org/tsconfig/strict.html)
- Type safety: narrowing and generics as first-class tools. (https://www.typescriptlang.org/docs/handbook/2/narrowing.html) (https://www.typescriptlang.org/docs/handbook/2/generics.html)
- Lint: ESLint + typescript-eslint. (https://typescript-eslint.io/getting-started/) (https://typescript-eslint.io/rules/)
- Format: Prettier. (https://prettier.io/docs/)
- Tests: Jest. (https://jestjs.io/docs/getting-started)
