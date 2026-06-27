from dataclasses import asdict

from para.proof_strategy_agent import AgentMemory, ProofStrategyAgent


def test_cold_start_agent_emits_verifier_obligation() -> None:
    agent = ProofStrategyAgent(AgentMemory())

    strategy, trace = agent.plan("overridesMethod", memory_enabled=False)

    assert strategy.target == "overridesMethod"
    assert strategy.strategy_id == "overridesMethod_cold_proof_strategy"
    assert trace.verifier_diagnostic.status == "needs_verifier_run"
    assert trace.decisions[-1].action == "VerifierDiagnostic"


def test_bounded_loop_records_refinement_trace() -> None:
    agent = ProofStrategyAgent(AgentMemory())

    loop = agent.plan_loop("canCallClass", memory_enabled=False, max_refinements=1)
    trace_payload = asdict(loop.final_trace)

    assert loop.stop_reason == "refined_strategy_ready_for_symbolic_admission"
    assert trace_payload["verifier_diagnostic"]["status"] == "ready_for_symbolic_admission"
    assert [step["action"] for step in trace_payload["decisions"]][-3:] == [
        "Reflect",
        "RefineStrategy",
        "VerifierDiagnosticAfterRefinement",
    ]
