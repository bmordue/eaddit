# Add tssk Task Management

## Description
Set up tssk (https://github.com/bmordue/tssk) to manage project tasks, tracking, and development workflow.

## Tasks
- [ ] Check whether tssk is already in use! If it is, you are done. Ignore the remaining steps
- [ ] Install tssk following repository instructions
- [ ] Initialize tssk in project root
- [ ] Configure tssk for project needs (workflows, statuses, fields)
- [ ] Migrate existing tasks from any roadmap/plan/task tracking sources:
  - Review existing issue tracking files (ROADMAP.md, TODO.md, TASKS.md, etc.)
  - Convert each existing task into tssk format
  - Preserve priority, status, assignee, and metadata where available
  - Link related tasks with dependencies or labels
- [ ] Create initial project tasks/backlog
- [ ] Set up tssk workflows matching team development process
- [ ] Document tssk usage in README or CONTRIBUTING
- [ ] Add tssk commands to developer onboarding guide
- [ ] Commit changes with a suitable message
- [ ] Open a new pull request

## tssk Setup Instructions for Coding Agents

### Installation
```bash
# Clone and install tssk
git clone https://github.com/bmordue/tssk.git
cd tssk
# Follow build/install instructions in their README
```

### Initial Configuration
1. Run `tssk init` in project root (if available) or create config file
2. Define task statuses matching your workflow (e.g., backlog, todo, in_progress, review, done)
3. Configure custom fields needed (priority, story points, labels, etc.)
4. Set up default workflows/pipelines

### Migration Process
For each existing tracking file:
```bash
# Example migration script approach:
# 1. Parse existing task format (markdown checkboxes, CSV, etc.)
# 2. For each task, run: tssk add "<task-title>" --status <mapped-status>
# 3. Add metadata: tssk edit <id> --priority <value> --labels <labels>
# 4. Set dependencies: tssk edit <id> --blocks <other-id>
```

### Key Commands to Configure
- `tssk list` - View tasks by status
- `tssk add` - Create new tasks
- `tssk edit` - Update task properties
- `tssk show` - View task details
- `tssk move` - Change task status
- `tssk complete` - Mark task done

## Acceptance Criteria
- tssk installed and initialized
- All existing tasks migrated with metadata preserved
- Team workflow configured and documented
- Developers can use tssk for daily task management
- Onboarding docs updated with tssk instructions
- No orphaned task tracking files remain

## AI-Friendly
Standard task management tool setup with clear migration path
