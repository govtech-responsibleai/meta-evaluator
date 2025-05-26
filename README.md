# Meta Evaluator

## Setup Instructions

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/govtech-responsibleai/meta-evaluator.git
   cd meta-evaluator
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up code quality tools (required to pass CI):**
   ```bash
   uv run pre-commit install
   ```

### Development Commands

- **Run linting:** `uv run ruff check`
- **Run formatting:** `uv run ruff format`
- **Run type checking:** `uv run pyright`
- **Run all quality checks:** `uv run pre-commit run --all-files`
