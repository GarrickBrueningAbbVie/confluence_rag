# Claude Code Rules for {{PROJECT_NAME}}
This file will be used for instructions for claude code to follow.
The main sections are as follows:
1. Code Style and Standards
2. Project Structure and Documentation
3. Development Checks and Git
4. Data Protection Rules
5. Data Science Applications

## Additional Context
For more detailed information, see:
- `.ai-context/` directory for structured context files
- `PROJECT_CONTEXT.md` for high-level project overview
- `docs/ai-working-guide.md` for comprehensive AI development guide
- `claude_sphinx_doc.md` for instructions on generating sphinx documentation
- `abbvie_style.md` for colors and fonts related to AbbVie brand styling

# Code Style and Standards
Follow these coding practices while developing or editing code.

## Formatting
- Use **Black** formatter with 100 character line length
- All code must be formatted with Black before committing
- Run `make format` to auto-format code

## Linting
- Follow **flake8** rules defined in `config/flake8.cfg`
- Maximum line length: 100 characters
- Run `make check-lint` to verify linting

## Type Checking
- All code must include **type hints** using Python's typing module
- Use **mypy** for static type checking
- Run `make check-types` to verify types
- Avoid using `Any` unless absolutely necessary

## Testing (Unit testing)
- All new code must include corresponding tests in `tests/` directory
- Use **pytest** for testing
- Test files should mirror source structure
- Run `make test` to execute tests
- Aim for comprehensive test coverage

## Test Scripts
Include stuff for testing scripts for data checking.

## Import Conventions
- Use absolute imports: `from {{PACKAGE_NAME}}.module import Class`
- Group imports: standard library, third-party, local
- Use `__init__.py` to expose public API

## Class and Function Structure
```python
from typing import Optional

class MyClass:
    """Class docstring describing purpose."""
    
    def __init__(self, param: str) -> None:
        """Initialize with type hints."""
        self.param = param
    
    def method(self, arg: int) -> Optional[str]:
        """Method with type hints and docstring."""
        pass
```

### Documentation
- Include well formated and readable docstrings for all public functions, classes, and modules
    - Doc strings must include the follow: Description, input, outputs at a minumum.
    - Include extra information as needed.
- Follow Google-style docstrings
- Document complex logic and non-obvious behavior
- If desired, create the sphinx code documentation at the end of the checks. 
    - Refer to the `claude_sphinx_doc.md` file for more detailed instructions.

## Error Handling
- Use appropriate exception types
- Provide meaningful error messages
- Follow existing error handling patterns in the codebase
- Create logging statements throughout the code
    - Logging statements must be informative
    - Loging statements should inform about the progress/status of pipelines and code
