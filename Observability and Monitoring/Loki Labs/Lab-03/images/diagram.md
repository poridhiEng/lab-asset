# Lab-03 Architecture Diagrams

Mermaid source for every architecture diagram referenced in `README.md`. Render to SVG with the Mermaid CLI (`mmdc -i diagram.md -o images/<n>.svg`) or paste into mermaid.live, then drop the SVGs into `images/` and the existing `README.md` `![](...)` references will pick them up.

---

## Diagram 1 — High-level overview

**Location in README:** line 5 (top of the doc, just under the introduction paragraph). Replaces `images/1.svg`.

**Purpose:** one-screen "what is this lab" picture — services on the left, observability stack in the middle, two consumer worlds (humans via Grafana, agents via logcli + Claude skills) on the right.

2

## Diagram 2 — Pipeline detail

**Location in README:** line 20 (end of "Task Description", right before Step 1). Replaces `images/2.svg`.

**Purpose:** zoomed-in view of the log pipeline — what each container produces, how Promtail discovers and labels them, what Loki indexes, and which two consumers read it back.

3

## Diagram 3 — Trace correlation across services (Scenario 3)

**Location in README:** suggested new placement at the start of "Scenario 3 — Trace one order across services" (around line 970+). Optional addition; the README does not currently reference it.

**Purpose:** make the `order_id` ↔ `trace_id` distinction visual — the linchpin of the cross-service stitching that Scenario 3 exercises.

4

## Diagram 4 — Skill orchestration inside `loki-investigate`

**Location in README:** suggested new placement inside "5e. `loki-investigate` — the playbook" (around line 836). Optional addition.

**Purpose:** show that `loki-investigate` is not a single CLI wrapper but a directed graph over the other three skills, ending in a cited diagnosis.

5
