# eaddit
Ingest a subreddit into a vector database that can be used for RAG queries

## Development Workflow

This project uses [tssk](https://github.com/bmordue/tssk) for task management and development workflow tracking.

### Key Commands

```bash
tssk list              # View all tasks
tssk list --status todo      # View tasks by status
tssk add -t "Title"    # Add a new task
tssk show <id>         # View task details
tssk status <id> <status>    # Update task status (todo, in-progress, done, blocked)
tssk deps add <id> <dep-id>  # Add task dependency
```

### Getting Started

1. Tasks are tracked in `.tssk.json` and `.tsks/` directory
2. Check available work: `tssk list --status todo`
3. Start working: `tssk status <id> in-progress`
4. Complete work: `tssk status <id> done`
