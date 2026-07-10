<!-- XYLEM:BEGIN v2 -->
## Xylem discipline

This machine runs the Xylem suite (context-keeper, agentsync, cambium). Follow this discipline:

- **At session start**, a context-keeper project summary is injected via hook. If a project summary was not injected at session start, pull `get_project_summary` before starting work. Treat any active constraints it lists as binding. Do not re-litigate settled decisions -- if one looks wrong, raise it via the agentsync mailbox or ask the user; don't silently override it.
- **Before multi-file or long-running work**, survey the agentsync board. Claim what you intend to touch. Never force over an active peer's claim without asking first. On claims running longer than ~15 minutes, post brief `update_status` notes at milestones -- 2 to 4 per claim, no more.
- **When you make an architectural or design decision**, record it in context-keeper with its rationale at the moment it's made -- not in a batch at the end.
- **On judgment calls with multiple defensible answers**, post the question to the agentsync mailbox and continue on non-dependent work rather than blocking.
- **Before ending**, record any new decisions, then release your claims with a closing note stating the outcome and the recommended next action. For a build session, the definition of done includes PUSHED TO ORIGIN, not just committed -- verify the push in the closing note.

Run `/xylem-discipline` to load the full workflow for a session.
<!-- XYLEM:END -->
