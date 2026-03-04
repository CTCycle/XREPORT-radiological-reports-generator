# Engineering and Python Standards

Mandatory standards for Python 3.14+ projects, covering backend services, FastAPI apps, and ML pipelines.

---

## 1. Python version and scope

- Target Python 3.14+ (aligned with `pyproject.toml`)
- Applies to:
  - Core libraries and services
  - FastAPI backends
  - ML and data pipelines
  - Tests, unless stated otherwise

---

## 2. Typing and correctness

### 2.1 Type rules

1. Use PEP 695 type parameters when applicable
2. Use built-in generics:
   - `list`, `dict`, `tuple`
   - Do not use `List`, `Dict`, `Tuple`
3. Use `|` unions, not `Optional` or `Union`
   - Example: `str | None`
4. Type hint:
   - All public APIs
   - Non-trivial internal logic
5. Import `Callable` from `collections.abc` only

### 2.2 Enforcement

- Static typing is mandatory but not a test replacement
- Enforce with mypy in CI

---

## 3. Imports

1. Imports must be top-level only
2. No conditional imports
3. Always use `collections.abc.Callable`
4. Use Keras 3.x directly, do not import TensorFlow via Keras

---

## 4. Code style and formatting

### 4.1 Tooling

- Style: PEP 8
- Formatter: Black or Ruff formatter
- Linter: Ruff
- Tests: pytest

### 4.2 Explicit rules

1. Use `os` for paths, not `pathlib`
2. Use `glob` only when justified
3. No leading underscores on variables, methods, or attributes
4. Use `self.name`, never `self._name`
5. Module filenames must be single words

---

## 5. Comments, docstrings, separators

### 5.1 Comments

- Minimal, factual, and necessary only

### 5.2 Docstrings

- Written only when explicitly requested
- Required sections:
  1. Summary
  2. Arguments
  3. Returns

### 5.3 Separators

- Classes: 79 `#`
- Functions and methods: `#` + 77 `-`
- No separator above `__init__`

---

## 6. Code structure and design

### 6.1 Principles

1. Single Responsibility Principle everywhere
2. Group related logic into cohesive modules
3. No nested class or function definitions
4. Separate logic from execution
5. Avoid over-abstraction
6. Prefer dependency injection or inversion of control

### 6.2 Object creation

- Use Factory, Builder, or Prototype when construction is complex

---

## 7. Architecture by system type

### 7.1 Frontend and UI

- MVC or MVVM
- Clear separation of rendering, state, and logic
- Thin controllers and views

### 7.2 Backend services

- Service Layer + Repository
- Business logic in services or domain classes
- Data access only via repositories or gateways

### 7.3 ML and data pipelines

- Pipeline, Factory, or Builder patterns
- Preprocessing, training, evaluation must be:
  - modular
  - reproducible
  - versioned

### 7.4 Event-driven systems

- Observer, Mediator, or Pub/Sub patterns

### 7.5 Plugins and configuration

- Strategy, Command, or Decorator patterns

### 7.6 Distributed systems

- CQRS, Saga, or Event Sourcing
- Use only when complexity justifies it

---

## 8. Testing

### 8.1 General rules

1. Arrange–Act–Assert
2. Readable and isolated tests
3. Mock dependencies for unit tests
4. Cover normal, edge, and failure cases

### 8.2 Test types

- Unit
- Integration
- Contract
- End-to-end

---

## 9. FastAPI standards

### 9.1 Application structure

1. Split endpoints into routers
2. Compose routers in the app
3. Keep modules cohesive and scalable

### 9.2 Dependency injection

- Centralize auth, authorization, DB sessions, and request-scoped resources

### 9.3 Validation and schemas

- Use Pydantic models and type hints
- Avoid manual validation
- Let schemas drive OpenAPI generation

### 9.4 Async usage

1. Use `async` only with fully non-blocking stacks
2. Never block inside async endpoints
3. Use async-compatible libraries if async is chosen
4. Prefer sync endpoints when async adds no value

### 9.5 Background work

1. Use `BackgroundTasks` for post-response work
2. Do not run CPU-heavy tasks in request handlers
3. Offload heavy workloads to workers or job queues

### 9.6 Testing FastAPI apps

1. Override dependencies in tests
2. Use consistent app initialization
3. Isolate shared state to avoid flaky tests

---

## 10. Tooling summary

- Formatter: Black or Ruff formatter
- Linter: Ruff
- Type checker: mypy
- Test runner: pytest
