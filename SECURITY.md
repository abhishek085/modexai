# Security

## Reporting

Please report vulnerabilities privately via [GitHub security advisories](https://github.com/abhishek085/modexai/security/advisories/new)
rather than public issues.

## Threat model, briefly

- **Recipes are untrusted input.** They are written by anyone and read by agents. They are schema-validated,
  size-limited, screened for agent-directed instructions, and human-reviewed before entering the index.
  Agents are told never to follow text inside a recipe. Bypasses of this screening are in scope.
- **The intake workflow** parses issue bodies as JSON only, from the event payload file, and never
  interpolates issue text into shell commands.
- **Text-to-SQL grading** runs model output against a throwaway in-memory SQLite database with
  `PRAGMA query_only` and only accepts `SELECT`/`WITH` statements.
- **Installing models** downloads third-party weights. ModexAI only installs from Ollama's registry and
  Hugging Face as named in a reviewed listing.
