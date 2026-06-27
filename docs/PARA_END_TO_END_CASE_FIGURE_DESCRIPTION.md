# PARA End-to-End Case Figure Description

This document describes the end-to-end case figure for PARA. The purpose of the
figure is to give reviewers one concrete, memorable walkthrough:

> a cross-level package dependency query becomes a verifier-admitted path
> strategy, then a SUPPORTED proof tree grounded in concrete project facts,
> while a matched negative control becomes INCONCLUSIVE.

The figure should not be a second architecture diagram. It should be a
case-driven proof walkthrough.

## Core Message

The figure should communicate four points at a glance:

1. **The query is cross-level.** It starts as a package-level architecture
   relation, but support must pass through class, method, call, and import
   evidence.
2. **The agent proposes a multi-predicate strategy.** The path-planning agent
   assembles the package/class/method/call/import path pattern.
3. **The verifier admits only the constrained strategy.** Symbolic admission
   promotes the call-and-import rule; weaker local or retrieval-only candidates
   are not sufficient under hard negatives.
4. **The output is auditable.** A positive query produces SUPPORTED with a proof
   tree and concrete EDB leaves; a matched negative control produces
   INCONCLUSIVE with zero bounded proofs.

## Recommended Title

Use a short title above the drawing:

```text
End-to-end case: cross-level package dependency proof
```

## Layout

Use a left-to-right layout with three visual bands:

```text
left: query, agent strategy, symbolic admission
middle: proof path over concrete typed entities
right: proof-status outputs and baseline contrast
```

The figure should be readable at double-column paper width. Avoid long fully
qualified Java names inside boxes. Use shortened labels in the drawing and let
the caption or text explain that the full contract contains concrete EDB facts.

## Left Band: Query and Strategy

Draw three stacked boxes on the left.

### Box 1: Query

Label:

```text
Query
packageCallDependency(Pa, Pb)
Pa: web.reactive.socket...
Pb: web.server
```

Purpose:

- shows the architecture-level question;
- keeps package names short;
- makes clear that the query starts at package level.

### Box 2: Agent ProofStrategy

Label:

```text
Agent ProofStrategy
package -> class -> method
call -> method -> class -> package
+ import guard
```

Purpose:

- shows that the agent proposes a path strategy, not an answer;
- highlights the multi-predicate composition;
- emphasizes the import guard as the extra constraint beyond call evidence.

### Box 3: Symbolic Admission

Label:

```text
Symbolic admission
seed0-2: 3/3 admitted
train F1 = 1.000
held-out F1 = 1.000
```

Purpose:

- shows that the strategy passed the verifier;
- connects the case to the three-seed experiment;
- makes the case empirical rather than purely illustrative.

Use an orange or strong bordered style for this box, because it is part of the
authority boundary.

## Middle Band: Proof Path

Draw the concrete proof path as a small graph or path tree. It should look like
an evidence chain, not like a generic pipeline.

Recommended nodes:

```text
Package Pa
Class HandshakeWebSocketService
Method createHandshakeInfo(...)
Method ServerWebExchange.getLogPrefix
Class ServerWebExchange
Package Pb
```

Recommended edges:

```text
Package Pa -> Class HandshakeWebSocketService
  label: containsClass

Class HandshakeWebSocketService -> Method createHandshakeInfo(...)
  label: containsMethod

Method createHandshakeInfo(...) -> Method ServerWebExchange.getLogPrefix
  label: callsMethod

Method ServerWebExchange.getLogPrefix -> Class ServerWebExchange
  label: containsMethod

Class ServerWebExchange -> Package Pb
  label: containsClass
```

Add a dashed guard edge:

```text
Class HandshakeWebSocketService --importsClass guard--> Class ServerWebExchange
```

This dashed guard is important. It visually explains why the admitted rule is
stronger than an import-only or call-only path.

## Admitted Rule Summary

Add a compact summary box below the proof path:

```text
Admitted rule combines
package/class containment + method call + import guard
Contract: 6 EDB leaves; 3 evidence chains
```

Do not put the full Horn clause in the figure unless there is enough space. The
full rule can remain in the caption or text:

```prolog
packageCallDependency(A,B) :-
    containsClass(A,V1),
    containsMethod(V1,V2),
    callsMethod(V2,V3),
    containsMethod(V4,V3),
    containsClass(B,V4),
    importsClass(V1,V4).
```

## Right Band: Outputs and Contrast

Draw three stacked boxes on the right.

### Box 1: SUPPORTED Output

Label:

```text
SUPPORTED output
finite proof tree
root: packageCallDependency
leaves: concrete EDB facts
```

Use green or blue styling. This box should look like the positive proof-status
output.

### Box 2: Negative Control

Label:

```text
Negative control
INCONCLUSIVE
0 proofs found
review obligation
```

Use gray styling. This box must not look like a false/negative proof. It is a
review obligation.

### Box 3: Baseline Contrast

Label:

```text
Same hard-negative protocol
Agent: 3/3, F1=1.000
GraphRAG: 1/3, F1=.855
Typed: 0/3 admitted
```

Use yellow or neutral highlight styling. This is a compact contrast, not the
main visual object. The main visual object should remain the proof path.

## Arrows

Use arrows with the following semantics:

- Query -> Agent ProofStrategy -> Symbolic Admission.
- Symbolic Admission -> Admitted Rule Summary.
- Admitted Rule Summary -> SUPPORTED output.
- Admitted Rule Summary -> Negative Control, using a dashed arrow.
- Proof path nodes should be connected by green solid arrows.
- Import guard should be a purple or dashed arrow.

Avoid arrows from the agent directly to SUPPORTED. The verifier and proof
contract must remain between strategy proposal and output.

## Caption Content

Use a caption with this content:

> End-to-end PARA case for a cross-level package dependency. The agent proposes
> a multi-predicate ProofStrategy; symbolic admission promotes only the
> call-and-import rule; bounded proof search returns a SUPPORTED proof tree with
> concrete EDB leaves. A negative control with no bounded positive proof becomes
> INCONCLUSIVE, and competing sources expose why the agent matters on this
> compositional target.

## Required Evidence Values

The figure should use these values, which are backed by the current experiment
artifacts:

- Target: `packageCallDependency`
- Agent: `3/3` admitted
- Agent train F1: `1.000`
- Agent held-out F1: `1.000`
- GraphRAG: `1/3` admitted
- GraphRAG admitted F1: `.855`
- Typed search: `0/3` admitted
- Representative positive contract: `SUPPORTED`
- Representative negative contract: `INCONCLUSIVE`
- Positive contract: `6` EDB leaves
- Positive contract: `3` path-evidence chains
- Negative control: `0` bounded proofs found

## What To Emphasize

The strongest visual emphasis should be:

1. the cross-level proof path;
2. the import guard;
3. symbolic admission before output;
4. SUPPORTED vs INCONCLUSIVE as different proof-status outcomes;
5. the compact baseline contrast.

If the drawing becomes crowded, shorten entity names before removing the
SUPPORTED/INCONCLUSIVE outputs or the baseline contrast.

## What To Avoid

Do not show:

- the agent directly producing SUPPORTED;
- GraphRAG or typed search as universally bad;
- INCONCLUSIVE as false;
- a generic label-classification pipeline;
- the full package/class/method identifiers if they make the diagram unreadable;
- proof-tree output without concrete EDB leaves;
- the baseline contrast as the main visual element.

## Safe Interpretation

Use this interpretation in surrounding text:

> The case does not claim that retrieval or typed search are weak in general.
> It shows that this cross-level package target requires a constrained
> call-and-import path. The path-planning agent proposes that strategy, the
> verifier admits it under hard negatives, and the proof engine materializes a
> fact-grounded SUPPORTED contract while preserving INCONCLUSIVE for missing
> support.
