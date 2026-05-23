"""
Agent Persistence: save / load full Physical AI agent state
===========================================================
Enables save/restore of a trained SelfImprovementLoop (or CuriousAgent)
including all learned state: Hebbian weights, causal graph, pattern memory,
long-term memory, prototype stores, and metadata.

Usage:
    from hdc.persistence import save_agent, load_agent
    save_agent(agent, "checkpoint.pt")
    agent2 = load_agent("checkpoint.pt", world_model_config)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


# ── What to save ──────────────────────────────────────────────────────────────

def _extract_state(agent) -> Dict[str, Any]:
    """Extract all serialisable state from a SelfImprovementLoop / CuriousAgent."""
    wm = agent.world.pipeline.world_model

    state: Dict[str, Any] = {
        "_version": "1.25",
        "_timestamp": time.time(),
        "_tick": agent._tick,
    }

    # 1. MultiHorizonPredictor weights (3 × D×D)
    mhp = wm.multi_horizon
    state["multi_horizon"] = {
        h.name: p.predictor.state_dict()
        for h, p in zip(mhp.horizons, mhp.predictors)
    }
    state["multi_horizon_error_buffers"] = {
        h.name: p.error_buffer.clone()
        for h, p in zip(mhp.horizons, mhp.predictors)
    }
    state["state_buffer"] = mhp.state_buffer.clone()

    # 2. ActionEvaluator prototypes
    ev = wm.action_evaluator
    state["safe_prototypes"]   = [p.clone() for p in ev._safe_prototypes]
    state["danger_prototypes"] = [p.clone() for p in ev._danger_prototypes]

    # 3. DigitalTwinSync history
    ts = wm.twin_sync
    state["twin_sync"] = {
        "divergence_history": list(ts._divergence_history[-1000:]),
        "total_steps": ts.total_steps,
        "n_recalibrations": ts.n_recalibrations,
    }

    # 4. CausalTransitionGraph (key-memory + accumulators)
    cg = agent.world.causal_graph
    if cg._key_mem._H_complex is not None:
        state["causal_graph"] = {
            "H_complex": cg._key_mem._H_complex.clone(),
            "labels": list(cg._key_mem._labels),
            "accum": [a.clone() for a in cg._next_accum],
            "counts": list(cg._next_count),
            "n_transitions": cg._n_transitions,
        }

    # 5. SequencePatternMemory (HoloGN complex memory)
    pm = agent.world.pattern_memory
    if pm._memory._H_complex is not None:
        state["pattern_memory"] = {
            "H_complex": pm._memory._H_complex.clone(),
            "labels": list(pm._memory._labels),
            "n_seen": list(pm._n_seen),
            "mem_labels": list(pm._labels),
            "n_patterns": pm._pattern_count,
        }

    # 6. HierarchicalContextEncoder EMA
    ctx = agent.world.context_encoder
    state["context_encoder"] = {
        "situation_proto": ctx._situation_proto.clone(),
        "sit_count": ctx._sit_count,
    }

    # 7. AutoCalibrator state
    cal = agent.calibrator
    state["calibrator"] = {
        "stable_streak": cal._stable_streak,
        "alarm_streak": cal._alarm_streak,
        "n_safe": cal._n_safe_registered,
        "n_danger": cal._n_danger_registered,
    }

    # 8. EnsembleUncertainty weights
    ens = agent.world.pipeline.ensemble
    state["ensemble"] = [m.predictor.state_dict() for m in ens._members]

    # 9. LongTermMemory (if present)
    if hasattr(agent, '_ltm') and agent._ltm is not None:
        ltm = agent._ltm
        entries = []
        for e in ltm.lt_memory._entries:
            entries.append({
                "sensor_hv": e.sensor_hv.clone(),
                "initial_surprise": e.initial_surprise,
                "final_error": e.final_error,
                "n_replays": e.n_replays,
            })
        state["ltm_entries"] = entries

    # 10. Improvement log summary
    if hasattr(agent, '_step_log'):
        state["improvement_summary"] = {
            "total_ticks": len(agent._step_log),
            "early_mean_error": sum(s.prediction_error for s in agent._step_log[:len(agent._step_log)//2]) / max(len(agent._step_log)//2, 1),
            "late_mean_error": sum(s.prediction_error for s in agent._step_log[len(agent._step_log)//2:]) / max(len(agent._step_log)//2, 1),
        }

    return state


def save_agent(agent, path: str):
    """
    Save full agent state to a .pt checkpoint file.

    Args:
        agent: SelfImprovementLoop or CuriousAgent
        path: File path for checkpoint
    """
    state = _extract_state(agent)
    torch.save(state, path)
    size_kb = Path(path).stat().st_size // 1024
    print(f"Saved agent checkpoint: {path} ({size_kb}KB, tick={agent._tick})")


def load_agent_state(agent, path: str) -> Dict[str, Any]:
    """
    Restore agent state from a checkpoint file.

    Args:
        agent: SelfImprovementLoop or CuriousAgent (already constructed)
        path: Checkpoint file path

    Returns:
        The loaded state dict (for inspection)
    """
    state = torch.load(path, map_location="cpu")
    wm = agent.world.pipeline.world_model
    mhp = wm.multi_horizon

    # Restore tick counter
    agent._tick = state.get("_tick", 0)

    # 1. Horizon predictor weights
    if "multi_horizon" in state:
        for h, p in zip(mhp.horizons, mhp.predictors):
            if h.name in state["multi_horizon"]:
                p.predictor.load_state_dict(state["multi_horizon"][h.name])
    if "multi_horizon_error_buffers" in state:
        for h, p in zip(mhp.horizons, mhp.predictors):
            if h.name in state["multi_horizon_error_buffers"]:
                p.error_buffer.copy_(state["multi_horizon_error_buffers"][h.name])
    if "state_buffer" in state:
        mhp.state_buffer.copy_(state["state_buffer"])

    # 2. Prototypes
    ev = wm.action_evaluator
    ev._safe_prototypes   = [p.clone() for p in state.get("safe_prototypes",   [])]
    ev._danger_prototypes = [p.clone() for p in state.get("danger_prototypes", [])]

    # 3. Twin sync
    ts_state = state.get("twin_sync", {})
    wm.twin_sync._divergence_history = list(ts_state.get("divergence_history", []))
    wm.twin_sync.total_steps         = ts_state.get("total_steps", 0)
    wm.twin_sync.n_recalibrations    = ts_state.get("n_recalibrations", 0)

    # 4. Causal graph
    cg = agent.world.causal_graph
    if "causal_graph" in state:
        cg_s = state["causal_graph"]
        cg._key_mem._H_complex = cg_s["H_complex"]
        cg._key_mem._labels    = list(cg_s["labels"])
        cg._next_accum = [a.clone() for a in cg_s["accum"]]
        cg._next_count = list(cg_s["counts"])
        cg._n_transitions = cg_s["n_transitions"]

    # 5. Pattern memory
    pm = agent.world.pattern_memory
    if "pattern_memory" in state:
        pm_s = state["pattern_memory"]
        pm._memory._H_complex = pm_s["H_complex"]
        pm._memory._labels    = list(pm_s["labels"])
        pm._n_seen    = list(pm_s["n_seen"])
        pm._labels    = list(pm_s["mem_labels"])
        pm._pattern_count = pm_s["n_patterns"]

    # 6. Context encoder
    if "context_encoder" in state:
        ctx = agent.world.context_encoder
        ctx._situation_proto.copy_(state["context_encoder"]["situation_proto"])
        ctx._sit_count = state["context_encoder"]["sit_count"]

    # 7. Calibrator
    if "calibrator" in state:
        cal = agent.calibrator
        cal._stable_streak      = state["calibrator"]["stable_streak"]
        cal._alarm_streak       = state["calibrator"]["alarm_streak"]
        cal._n_safe_registered  = state["calibrator"]["n_safe"]
        cal._n_danger_registered = state["calibrator"]["n_danger"]

    # 8. Ensemble
    if "ensemble" in state:
        ens = agent.world.pipeline.ensemble
        for m, sd in zip(ens._members, state["ensemble"]):
            m.predictor.load_state_dict(sd)

    ver = state.get("_version", "?")
    tick = state.get("_tick", 0)
    ts   = state.get("_timestamp", 0)
    import datetime
    dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
    print(f"Loaded checkpoint v{ver} (tick={tick}, saved {dt})")
    return state


# ── Generic save / load for any SNNTraining model ──────────────────────────────

def save_model(obj: Any, path: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Generic checkpoint: save any Python/PyTorch object to disk.

    Works with:
      - WorldModelStream (snntraining.stream)
      - EliteSNNTrainingModel, EliteSNNTrainingPipeline
      - Any nn.Module with state_dict()
      - Any Python object (pickle fallback)

    Args:
        obj:      Model or pipeline to save
        path:     Target file path (.pt recommended)
        metadata: Optional dict of extra metadata (version, timestamp, etc.)
    """
    payload: Dict[str, Any] = {
        "_snntraining_checkpoint": True,
        "_timestamp": time.time(),
        "_metadata": metadata or {},
        "_classname": type(obj).__name__,
    }
    if hasattr(obj, "state_dict"):
        payload["state_dict"] = obj.state_dict()
        payload["_save_type"] = "state_dict"
    elif hasattr(obj, "save"):
        # WorldModelStream uses .save(path)
        obj.save(path)
        return
    else:
        payload["_object"] = obj
        payload["_save_type"] = "pickle"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    size_kb = Path(path).stat().st_size // 1024
    print(f"Saved {type(obj).__name__} → {path} ({size_kb} KB)")


def load_model(obj: Any, path: str) -> Dict[str, Any]:
    """
    Load a checkpoint saved with save_model().

    Applies state_dict if available, otherwise returns the pickled object.

    Args:
        obj:  Existing model instance to load weights into (or None)
        path: Checkpoint file path

    Returns:
        The full payload dict.
    """
    payload = torch.load(path, map_location="cpu")

    if not payload.get("_snntraining_checkpoint"):
        # Try as state_dict directly
        if obj is not None and hasattr(obj, "load_state_dict"):
            obj.load_state_dict(payload)
        return payload

    save_type = payload.get("_save_type", "state_dict")
    if save_type == "state_dict" and obj is not None and hasattr(obj, "load_state_dict"):
        obj.load_state_dict(payload["state_dict"])
        cn  = payload.get("_classname", "?")
        ts  = payload.get("_timestamp", 0)
        import datetime
        dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
        print(f"Loaded {cn} checkpoint (saved {dt})")

    return payload
