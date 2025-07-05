# create-pr - Comprehensive PR Creation with Issue Linking

Creates a pull request using the project's PR template, automatically investigates and links related open issues, and handles proper branch management with commit and push operations.

## Usage
/project:create-pr

## What it does
1. Investigates open issues to identify potentially related ones
2. Generates and switches to an appropriately named feature branch
3. Commits current changes with descriptive messaging
4. Pushes the branch to remote repository
5. Creates a PR using the project's template
6. Automatically links related issues when identified

## Implementation

Execute a comprehensive PR creation workflow that handles branch management, issue investigation, and template-based PR creation.

Process:
1. Use `gh issue list --state open` to investigate potentially related open issues
2. Create a feature branch following the pattern: `feature/[descriptive-name]`
3. Stage and commit current changes with an appropriate commit message
4. Push the branch to the remote repository with upstream tracking
5. Create a pull request using the project's PR template (`.github/PULL_REQUEST_TEMPLATE.md`)
6. When related issues are identified, automatically link them using "Closes #123" or "Relates to #123" syntax
7. Create the PR in draft status initially to allow for final review before marking as ready

Ensure proper error handling throughout the process and provide clear feedback on each step's completion status.
