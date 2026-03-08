# Coding Philosophy & Design Principles

## Architecture & Design

- **Functional-first**: Prefer the functional approach by default.
- **Classes as abstractions**: Use classes when many related functions belong together (e.g., a `Car` with many behaviors). Do not use deep inheritance — keep inheritance depth shallow.
- **Abstract classes & protocols**: Use these to define available functionality/contracts for a class.
- **Mixins over base inheritance**: Favor mixins for sharing functionality across classes.
- **Modular & injectable**: Prefer modularity and dependency injection over embedding services directly in a class. Separation leads to easier code management.

## Control Flow

- Prefer `match/case` where possible.
- `if/else` is fine — keep conditions tidy and readable.
- Prefer loops over direct recursion for readability. Recursive algorithms like divide-and-conquer are acceptable where the algorithm calls for it.

## Performance & Resources

- Readability is sometimes more important than performance gains.
- Performance only becomes a priority when it starts to affect the overall system.
- Think generally about keeping resource usage low (memory and CPU).
- Use **generators** to avoid loading large lists into memory.
- **`itertools`** is preferred for iteration patterns.
- Use **fixed arrays** when the number of items N is known and fixed.

## Linting

- **Tool**: `ruff>=0.4` (configured in `pyproject.toml`)
- **Run**: `python -m ruff check .` / `python -m ruff check . --fix` for auto-fixes
- **Rules**: `E` (line length/whitespace), `W` (trailing whitespace), `F` (unused imports/undefined names), `ANN` (missing type annotations), `I` (import ordering)
- **Line limit**: 88 characters
- `tests/*` exempt from `ANN` rules

## Typing

- Use typing as strictly as you would in a statically typed language.
- Every function should clearly annotate what it takes and what it returns.
- Developers must never have to guess the shape of inputs or outputs.

## Classes & Encapsulation

- Never leak or export private methods of a class.
- Use class validation libraries like **Pydantic** to manage and surface errors cleanly.

## Error Handling

- Follow **Go-style error returns**: return the value and an error if one occurred.
  ```python
  def get_user(id: int) -> tuple[User | None, Exception | None]:
      ...
  ```
- This makes error handling explicit and puts responsibility on the caller.

## Comments & Documentation

- Keep the summary comment as concise as a tweet (≤280 chars).
- You may add examples of the function being called, with arguments and return values shown.
- Keep it concise — write for other developers, not for documentation generators.

## Testing

- Use **mocking** when the underlying function has already been thoroughly tested.
- Untested functions should be examined carefully before mocking.
- Use **parameterized tests** to group test cases that share the same structure and assertions.

## File Organization

- Keep files small and focused.
- If a file (including test files) starts getting too long, break it up into its own files.
- **Modular functions first**: Prefer standalone modular functions by default. When the number of related utility functions grows unmanageable, consolidate them into a **static utility class** (no instantiation, all `@staticmethod`) — similar to how PyTorch uses `torch.nn.functional`. This keeps the namespace clean and groups related operations without forcing OOP overhead.

## Decorators

- Use decorators when you have a utility that wraps other methods — e.g., logging, retry logic, auth checks.
- Good decorators are reusable and cross-cutting, not business-logic-specific.
