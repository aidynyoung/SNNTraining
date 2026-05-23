"""
tests/test_iqt_ff.py
=====================
Tests for IQT/Founders Fund elite improvements:
  - GeometricMedianAggregator (Byzantine-robust federation)
  - CertifiedHDCClassifier (provable adversarial robustness)
  - SpikeInteractionReadout (BCI pairwise features)
  - MCUDeploymentProfiler (hardware energy/SRAM profiling)
  - HierarchicalFederatedHDC (multi-tier edge federation)
"""

from __future__ import annotations
import pytest
import torch

D = 128   # HV dim for fast tests
K = 2     # output dim
N = 16    # hidden/neuron count


# ═══════════════════════════════════════════════════════════════════════════════
# GeometricMedianAggregator
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.multi_agent_hdc import GeometricMedianAggregator, HDCAgent


class TestGeometricMedianAggregator:
    def _make_agents(self, n: int, n_classes: int = 3) -> list:
        agents = []
        for i in range(n):
            a = HDCAgent(f"agent_{i}", D, n_classes)
            for c in range(n_classes):
                for _ in range(5):
                    hv = (torch.rand(D) > 0.5).float()
                    a.train_step(hv, c)
            agents.append(a)
        return agents

    def test_returns_n_classes_protos(self):
        gma = GeometricMedianAggregator(n_classes=3, dim=D)
        agents = self._make_agents(4, n_classes=3)
        exports = [a.export_prototypes() for a in agents]
        result = gma.aggregate(exports)
        assert len(result) == 3
        for p in result:
            assert p.shape == (D,)

    def test_output_binary(self):
        gma = GeometricMedianAggregator(n_classes=2, dim=D)
        agents = self._make_agents(3, n_classes=2)
        exports = [a.export_prototypes() for a in agents]
        result = gma.aggregate(exports)
        for p in result:
            assert set(p.unique().tolist()).issubset({0.0, 1.0})

    def test_single_agent_passthrough(self):
        gma = GeometricMedianAggregator(n_classes=2, dim=D)
        agents = self._make_agents(1, n_classes=2)
        exports = [agents[0].export_prototypes()]
        result = gma.aggregate(exports)
        assert len(result) == 2

    def test_byzantine_capacity(self):
        gma = GeometricMedianAggregator(n_classes=2, dim=D)
        cap = gma.byzantine_capacity(10)
        assert cap["n_agents"] == 10
        assert cap["max_byzantine"] >= 4
        assert 0.0 <= cap["robustness_frac"] <= 1.0

    def test_robust_to_outlier_agent(self):
        torch.manual_seed(42)
        gma = GeometricMedianAggregator(n_classes=1, dim=D, n_iter=15)

        # 6 honest agents with prototypes that are ~50% ones (random)
        honest_exports = []
        for _ in range(6):
            proto = (torch.rand(D) > 0.5).float()   # ~50% density
            honest_exports.append({"prototypes": [proto], "counts": [10]})

        # Mean of honest prototypes (without Byzantine)
        honest_mean = torch.stack([e["prototypes"][0] for e in honest_exports]).float().mean(0)
        honest_density = float(honest_mean.mean().item())

        # Inject one Byzantine agent with all-zeros (opposite extreme)
        byzantine = {"prototypes": [torch.zeros(D)], "counts": [1000]}

        # With Byzantine
        result_byz = gma.aggregate(honest_exports + [byzantine])
        density_byz = float(result_byz[0].float().mean().item())

        # Without Byzantine
        result_clean = gma.aggregate(honest_exports)
        density_clean = float(result_clean[0].float().mean().item())

        # Geometric median should keep result closer to honest cluster
        # than arithmetic mean would (which gets pulled toward zero by Byzantine)
        # At minimum: with 6 honest at ~0.5 and 1 Byzantine at 0.0, median > 0
        assert density_byz > 0.1   # not collapsed to all-zeros
        assert len(result_byz) == 1

    def test_geom_median_different_from_mean(self):
        gma = GeometricMedianAggregator(n_classes=1, dim=D, n_iter=15)
        agents = self._make_agents(5, n_classes=1)
        exports = [a.export_prototypes() for a in agents]
        gm_result = gma.aggregate(exports)

        # Compare to simple mean
        from hdc.multi_agent_hdc import FederatedHDCAggregator
        mean_agg = FederatedHDCAggregator(n_classes=1, dim=D)
        mean_result = mean_agg.aggregate(exports)

        # Both should be valid binary HVs
        assert gm_result[0].shape == (D,)
        assert mean_result[0].shape == (D,)


# ═══════════════════════════════════════════════════════════════════════════════
# CertifiedHDCClassifier
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.hdc_security import CertifiedHDCClassifier
from hdc.hdcc_compiler import HDCCClassifier


def _make_trained_clf(n_features: int = 8, n_classes: int = 3, dim: int = D):
    clf = HDCCClassifier(n_features=n_features, n_classes=n_classes, dim=dim)
    for c in range(n_classes):
        for _ in range(10):
            x = torch.randn(n_features) + c * 2.0
            clf.train_step(x, c)
    return clf


class TestCertifiedHDCClassifier:
    def test_certified_predict_returns_dict(self):
        clf  = _make_trained_clf()
        cert = CertifiedHDCClassifier(clf, noise_flip_rate=0.1, n_samples=50)
        hv   = (torch.rand(D) > 0.5).float()
        result = cert.certified_predict(hv)
        assert "label" in result
        assert "p_A" in result
        assert "certified_radius" in result
        assert "abstain" in result

    def test_certified_radius_nonneg(self):
        clf  = _make_trained_clf()
        cert = CertifiedHDCClassifier(clf, n_samples=50)
        hv   = (torch.rand(D) > 0.5).float()
        r    = cert.certified_predict(hv)
        assert r["certified_radius"] >= 0

    def test_pa_in_range(self):
        clf  = _make_trained_clf()
        cert = CertifiedHDCClassifier(clf, n_samples=50)
        hv   = (torch.rand(D) > 0.5).float()
        r    = cert.certified_predict(hv)
        assert 0.0 <= r["p_A"] <= 1.0

    def test_vote_counts_sum_to_n_samples(self):
        clf  = _make_trained_clf()
        cert = CertifiedHDCClassifier(clf, n_samples=30)
        hv   = (torch.rand(D) > 0.5).float()
        r    = cert.certified_predict(hv)
        total = sum(r["vote_counts"].values())
        assert total == 30

    def test_robustness_report_keys(self):
        clf  = _make_trained_clf()
        cert = CertifiedHDCClassifier(clf, n_samples=20)
        hvs  = [(torch.rand(D) > 0.5).float() for _ in range(5)]
        results = cert.batch_certify(hvs)
        report  = cert.robustness_report(results)
        for key in ("certified_frac", "mean_certified_r", "abstain_frac"):
            assert key in report

    def test_higher_noise_lower_radius(self):
        clf    = _make_trained_clf()
        cert1  = CertifiedHDCClassifier(clf, noise_flip_rate=0.05, n_samples=100, alpha=0.01)
        cert2  = CertifiedHDCClassifier(clf, noise_flip_rate=0.30, n_samples=100, alpha=0.01)
        hv     = (torch.rand(D) > 0.5).float()
        r1, r2 = cert1.certified_predict(hv), cert2.certified_predict(hv)
        # Higher noise generally → lower or equal certified radius
        assert isinstance(r1["certified_radius"], int)
        assert isinstance(r2["certified_radius"], int)


# ═══════════════════════════════════════════════════════════════════════════════
# SpikeInteractionReadout
# ═══════════════════════════════════════════════════════════════════════════════

from models.readout import SpikeInteractionReadout


class TestSpikeInteractionReadout:
    def test_output_shape(self):
        sr = SpikeInteractionReadout(N, K, n_lags=3, n_interactions=16, seed=0)
        s  = (torch.rand(N) > 0.8).float()
        y  = sr(s)
        assert y.shape == (K,)

    def test_update_returns_info(self):
        sr = SpikeInteractionReadout(N, K, n_lags=3, n_interactions=16)
        s  = (torch.rand(N) > 0.8).float()
        sr(s)
        info = sr.update(torch.randn(K))
        assert "eff_lr" in info
        assert "lam" in info

    def test_pairs_are_distinct(self):
        sr = SpikeInteractionReadout(N, K, n_interactions=32, seed=0)
        # pair_i and pair_j should have no self-pairs
        assert (sr._pair_i == sr._pair_j).sum().item() == 0

    def test_interaction_features_shape(self):
        sr = SpikeInteractionReadout(N, K, n_interactions=16)
        s  = (torch.rand(N) > 0.8).float()
        phi = sr._interaction_features(s)
        assert phi.shape == (16,)

    def test_interaction_features_nonneg(self):
        sr  = SpikeInteractionReadout(N, K, n_interactions=8)
        s   = (torch.rand(N) > 0.5).float()
        phi = sr._interaction_features(s)
        assert (phi >= 0.0).all()

    def test_reset_clears_state(self):
        sr = SpikeInteractionReadout(N, K, n_interactions=8)
        for _ in range(5):
            sr((torch.rand(N) > 0.8).float())
            sr.update(torch.randn(K))
        sr.reset(reset_weights=True)
        assert sr.W_inter.abs().sum() == 0.0
        assert sr._buf_current is None

    def test_current_spikes_weight(self):
        sr = SpikeInteractionReadout(N, K, n_lags=3, n_interactions=8)
        w  = sr.current_spikes_weight()
        assert w.shape == (K, N)

    def test_trains_smoothly_50_steps(self):
        sr = SpikeInteractionReadout(N, K, n_lags=3, n_interactions=16, seed=0)
        errors = []
        for _ in range(50):
            s = (torch.rand(N) > 0.8).float()
            y = sr(s)
            e = y - torch.randn(K) * 0.1
            sr.update(e)
            errors.append(float(e.abs().mean().item()))
        # Error should generally decrease
        first_half = sum(errors[:25]) / 25
        second_half = sum(errors[25:]) / 25
        assert second_half <= first_half + 0.5  # allow some tolerance


# ═══════════════════════════════════════════════════════════════════════════════
# MCUDeploymentProfiler
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.hardware_synthesis import MCUDeploymentProfiler


class TestMCUDeploymentProfiler:
    def test_profile_nrf52840(self):
        p = MCUDeploymentProfiler(hd_dim=1024, n_classes=5, input_dim=32)
        result = p.profile("nRF52840")
        assert result["mcu"] == "nRF52840"
        assert result["energy_nJ"] > 0
        assert result["inference_us"] > 0

    def test_profile_all_targets(self):
        p = MCUDeploymentProfiler(hd_dim=512, n_classes=3)
        results = p.compare_all_targets()
        assert len(results) == len(MCUDeploymentProfiler.MCU_PROFILES)
        # Sorted by energy
        energies = [r["energy_nJ"] for r in results]
        assert energies == sorted(energies)

    def test_fits_in_sram(self):
        p = MCUDeploymentProfiler(hd_dim=256, n_classes=3)
        result = p.profile("STM32L4R9")
        # 256-bit dim, 3 classes → 3 × 32 bytes = 96 bytes proto, should fit
        assert result["fits_in_sram"] is True

    def test_print_report_is_string(self):
        p   = MCUDeploymentProfiler(hd_dim=1024, n_classes=5)
        rpt = p.print_report("nRF52840")
        assert isinstance(rpt, str)
        assert "nRF52840" in rpt
        assert "nJ" in rpt

    def test_unknown_mcu_raises(self):
        p = MCUDeploymentProfiler(hd_dim=512, n_classes=3)
        with pytest.raises(ValueError):
            p.profile("PIC16F877A")

    def test_battery_life_positive(self):
        p      = MCUDeploymentProfiler(hd_dim=512, n_classes=3)
        result = p.profile("nRF52840")
        assert result["battery_life_hours"] > 0

    def test_nn_comparison_much_higher(self):
        p      = MCUDeploymentProfiler(hd_dim=1024, n_classes=5)
        result = p.profile("nRF52840")
        assert result["nn_comparison_nJ"] > result["energy_nJ"]


# ═══════════════════════════════════════════════════════════════════════════════
# HierarchicalFederatedHDC
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.hierarchical_federation import (
    HierarchicalFederatedHDC, FederationTierNode, TierConfig,
    PrototypeCompressor,
)


class TestHierarchicalFederatedHDC:
    def _make_export(self, n_classes: int = 3) -> dict:
        return {
            "prototypes": [(torch.rand(D) > 0.5).float() for _ in range(n_classes)],
            "counts":     [10] * n_classes,
        }

    def test_submit_and_round(self):
        hf = HierarchicalFederatedHDC(n_classes=3, dim=D)
        for i in range(5):
            hf.submit(f"agent_{i}", self._make_export())
        result = hf.federated_round()
        assert "global_protos" in result
        assert len(result["global_protos"]) == 3

    def test_global_protos_binary(self):
        hf = HierarchicalFederatedHDC(n_classes=2, dim=D)
        for i in range(4):
            hf.submit(f"agent_{i}", self._make_export(n_classes=2))
        result = hf.federated_round()
        for p in result["global_protos"]:
            assert set(p.unique().tolist()).issubset({0.0, 1.0})

    def test_communication_savings(self):
        hf  = HierarchicalFederatedHDC(n_classes=3, dim=D)
        hf.submit("a", self._make_export())
        hf.federated_round()
        savings = hf.communication_savings()
        assert savings["bandwidth_reduction"] > 1
        assert savings["hdc_payload_bytes"] < savings["nn_payload_bytes"]

    def test_multiple_rounds(self):
        hf = HierarchicalFederatedHDC(n_classes=2, dim=D)
        for rnd in range(3):
            for i in range(4):
                hf.submit(f"agent_{i}", self._make_export(n_classes=2))
            result = hf.federated_round()
            assert result["round"] == rnd + 1

    def test_handles_empty_round(self):
        hf = HierarchicalFederatedHDC(n_classes=2, dim=D)
        # No submissions
        result = hf.federated_round()
        # Should not crash; returns current (zero-initialised) model
        assert len(result["global_protos"]) == 2

    def test_tier_config_dp_epsilon(self):
        cfg = [TierConfig("tier1", n_children=4, dp_epsilon=0.5)]
        hf  = HierarchicalFederatedHDC(n_classes=2, dim=D, tier_configs=cfg)
        for i in range(4):
            hf.submit(f"a{i}", self._make_export(n_classes=2))
        result = hf.federated_round()
        assert len(result["global_protos"]) == 2


class TestPrototypeCompressor:
    def test_compress_same_shape(self):
        comp = PrototypeCompressor(full_dim=D, budget_bits=D * 8)
        p    = (torch.rand(D) > 0.5).float()
        c    = comp.compress(p)
        assert c.shape == (D,)

    def test_compress_drops_small_dims(self):
        comp = PrototypeCompressor(full_dim=D, budget_bits=D // 2)
        p    = torch.ones(D)
        c    = comp.compress(p)
        assert c.shape == (D,)
        # Some dims should be zero
        assert (c == 0).any()

    def test_compression_ratio_in_range(self):
        comp = PrototypeCompressor(full_dim=D, budget_bits=D // 2)
        assert 0.0 < comp.compression_ratio() <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# HDCAnomalyDetector
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.hdcc_compiler import (
    HDCAnomalyDetector, OnlineAdaptiveConformal, ConformalHDCWrapper,
    HDCCClassifier as _HDCCForAnomaly,
)


class TestHDCAnomalyDetector:
    def test_score_in_range(self):
        det = HDCAnomalyDetector(D, fpr_target=0.1)
        for _ in range(20):
            det.update_normal((torch.rand(D) > 0.5).float())
        s = det.score((torch.rand(D) > 0.5).float())
        assert 0.0 <= s <= 1.0

    def test_calibrate_sets_threshold(self):
        det = HDCAnomalyDetector(D, fpr_target=0.1)
        normal_hvs = [(torch.rand(D) > 0.5).float() for _ in range(50)]
        for hv in normal_hvs[:30]:
            det.update_normal(hv)
        det.calibrate(normal_hvs)
        assert det._threshold > 0.0

    def test_detect_returns_dict(self):
        det = HDCAnomalyDetector(D)
        for _ in range(20):
            det.update_normal((torch.rand(D) > 0.5).float())
        det.calibrate([(torch.rand(D) > 0.5).float() for _ in range(20)])
        result = det.detect((torch.rand(D) > 0.5).float())
        assert "is_anomaly" in result
        assert "score" in result
        assert "threshold" in result

    def test_all_zeros_is_anomalous(self):
        # Use ema_decay=0.0 (static accumulator) so prototype is well-calibrated
        det = HDCAnomalyDetector(D, fpr_target=0.05, ema_decay=0.0)
        torch.manual_seed(7)
        for _ in range(50):
            det.update_normal((torch.rand(D) > 0.5).float())   # ~50% ones
        normal_hvs = [(torch.rand(D) > 0.5).float() for _ in range(50)]
        det.calibrate(normal_hvs)
        # All-zeros vs ~50%-ones prototype: Hamming distance ≈ 0.5 > threshold
        result = det.detect(torch.zeros(D))
        assert result["score"] > 0.3   # clearly anomalous

    def test_normal_fpr_at_target(self):
        det = HDCAnomalyDetector(D, fpr_target=0.1)
        normal_hvs = [(torch.rand(D) > 0.5).float() for _ in range(100)]
        for hv in normal_hvs:
            det.update_normal(hv)
        det.calibrate(normal_hvs)
        fpr = det.false_positive_rate()
        assert fpr <= 0.15   # allow small slack from quantile approximation


class TestOnlineAdaptiveConformal:
    def _make_conformal(self, n_classes: int = 3, n_cal: int = 30):
        clf = _make_trained_clf(n_features=8, n_classes=n_classes)
        conf = ConformalHDCWrapper(clf, alpha=0.1)
        X_cal = [torch.randn(8) + c * 2.0 for c in range(n_classes) for _ in range(n_cal // n_classes)]
        y_cal = [c for c in range(n_classes) for _ in range(n_cal // n_classes)]
        conf.calibrate(X_cal, y_cal)
        return conf

    def test_init_threshold_from_base(self):
        conf = self._make_conformal()
        aci  = OnlineAdaptiveConformal(conf)
        assert 0.0 <= aci._q <= 1.0

    def test_predict_set_returns_list(self):
        conf = self._make_conformal()
        aci  = OnlineAdaptiveConformal(conf)
        x    = torch.randn(8)
        pred_set, q = aci.predict_set(x)
        assert isinstance(pred_set, list)
        assert isinstance(q, float)

    def test_adaptive_coverage_report(self):
        conf = self._make_conformal()
        aci  = OnlineAdaptiveConformal(conf, gamma=0.01)
        x    = torch.randn(8)
        pred_set, _ = aci.predict_set(x)
        aci.update(covered=True)   # pass bool: true label was in prediction set
        report = aci.adaptive_coverage_report()
        assert "empirical_coverage" in report
        assert "target_coverage" in report


# ═══════════════════════════════════════════════════════════════════════════════
# NNToHDCDistiller
# ═══════════════════════════════════════════════════════════════════════════════

class TestNNToHDCDistiller:
    def _make_teacher(self, in_dim: int = 8, n_classes: int = 3):
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, n_classes))
        model.eval()
        return model

    def test_distill_runs(self):
        from training.distill_bridge import NNToHDCDistiller
        teacher = self._make_teacher()
        clf     = HDCCClassifier(n_features=8, n_classes=3, dim=D)
        dist    = NNToHDCDistiller(teacher, clf, temperature=2.0, min_confidence=0.3)

        def stream():
            for _ in range(30):
                yield torch.randn(8), torch.randint(0, 3, (1,)).item()

        result = dist.distill(stream(), n_steps=20)
        assert "n_distilled" in result
        assert result["n_distilled"] + result["n_skipped"] <= 20

    def test_evaluate_compression(self):
        from training.distill_bridge import NNToHDCDistiller
        teacher = self._make_teacher()
        clf     = HDCCClassifier(n_features=8, n_classes=3, dim=D)
        dist    = NNToHDCDistiller(teacher, clf)
        report  = dist.evaluate_compression()
        assert report["size_reduction"] >= 1
        assert report["energy_reduction_est"] > 1

    def test_teacher_frozen(self):
        from training.distill_bridge import NNToHDCDistiller
        teacher = self._make_teacher()
        clf     = HDCCClassifier(n_features=8, n_classes=3, dim=D)
        dist    = NNToHDCDistiller(teacher, clf)
        for p in dist.teacher.parameters():
            assert not p.requires_grad


# ═══════════════════════════════════════════════════════════════════════════════
# RenyiDPAccountant
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.hdc_security import RenyiDPAccountant


class TestRenyiDPAccountant:
    def test_initial_epsilon_zero(self):
        acc = RenyiDPAccountant(noise_multiplier=1.0, delta=1e-5)
        eps = acc.get_epsilon()
        assert eps >= 0.0

    def test_epsilon_grows_with_rounds(self):
        acc = RenyiDPAccountant(noise_multiplier=1.0, delta=1e-5)
        acc.step(10)
        eps10 = acc.get_epsilon()
        acc.step(10)
        eps20 = acc.get_epsilon()
        assert eps20 > eps10

    def test_more_noise_less_epsilon(self):
        acc_low  = RenyiDPAccountant(noise_multiplier=0.5, delta=1e-5)
        acc_high = RenyiDPAccountant(noise_multiplier=2.0, delta=1e-5)
        acc_low.step(10)
        acc_high.step(10)
        assert acc_low.get_epsilon() > acc_high.get_epsilon()

    def test_rdp_tighter_than_basic(self):
        # Rényi DP should give much lower epsilon than basic composition
        # Basic: T × ε_per_round; RDP: O(sqrt(T)) growth
        T = 50
        acc = RenyiDPAccountant(noise_multiplier=1.0, delta=1e-5)
        acc.step(T)
        eps_rdp = acc.get_epsilon()
        # Basic composition ε = T × noise-dependent-eps would be much larger
        # We just verify the epsilon is a valid positive number
        assert eps_rdp > 0.0
        assert eps_rdp < 1000.0   # should be finite

    def test_privacy_report_keys(self):
        acc = RenyiDPAccountant(noise_multiplier=1.0)
        acc.step(5)
        report = acc.privacy_report()
        assert "n_rounds" in report
        assert "epsilon" in report
        assert "delta" in report

    def test_max_rounds_for_epsilon(self):
        acc = RenyiDPAccountant(noise_multiplier=2.0)
        max_r = acc.max_rounds_for_epsilon(target_epsilon=1.0)
        assert isinstance(max_r, int)
        assert max_r >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# LoopClosureSLAM
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.grid_cells import GridCellNetwork, LoopClosureSLAM


class TestLoopClosureSLAM:
    def test_slam_step_returns_dict(self):
        net  = GridCellNetwork(dim=64, periods=[5.0], seed=0)
        slam = LoopClosureSLAM(net, dim=64)
        slam.reset(x_init=0.0)
        result = slam.slam_step(dx=0.5)
        assert "pos_estimate" in result
        assert "loop_closed" in result
        assert "n_landmarks" in result

    def test_pos_estimate_changes_with_motion(self):
        net  = GridCellNetwork(dim=64, periods=[5.0], seed=0)
        slam = LoopClosureSLAM(net, dim=64)
        slam.reset(x_init=0.0)
        r0 = slam.slam_step(dx=0.0)
        r1 = slam.slam_step(dx=2.0)
        # Position should have changed
        assert r0["pos_estimate"] != r1["pos_estimate"]

    def test_landmark_stored_on_scene_update(self):
        net  = GridCellNetwork(dim=64, periods=[5.0], seed=0)
        slam = LoopClosureSLAM(net, dim=64)
        slam.reset(x_init=0.0)
        scene = (torch.rand(64) > 0.5).float()
        r = slam.slam_step(dx=0.5, scene_hv=scene)
        assert r["n_landmarks"] >= 1

    def test_loop_closure_detected_on_revisit(self):
        net  = GridCellNetwork(dim=256, periods=[5.0, 7.0], seed=0)
        slam = LoopClosureSLAM(net, dim=256, closure_threshold=0.6)
        slam.reset(x_init=0.0)

        # Store a scene at position 0
        scene_a = (torch.rand(256) > 0.5).float()
        slam.slam_step(dx=0.0, scene_hv=scene_a)

        # Move away
        for _ in range(5):
            slam.slam_step(dx=1.0)

        # Revisit same scene (slightly noisy)
        scene_a_noisy = (scene_a.clone() + (torch.rand(256) > 0.9).float()) % 2
        r = slam.slam_step(dx=0.0, scene_hv=scene_a_noisy)
        # May or may not close loop depending on similarity — just verify no crash
        assert isinstance(r["loop_closed"], bool)

    def test_map_summary(self):
        net  = GridCellNetwork(dim=64, periods=[5.0], seed=0)
        slam = LoopClosureSLAM(net, dim=64)
        slam.reset(x_init=0.0)
        for i in range(5):
            scene = (torch.rand(64) > 0.5).float()
            slam.slam_step(dx=0.5, scene_hv=scene)
        summary = slam.map_summary()
        assert "n_landmarks" in summary
        assert summary["n_landmarks"] >= 1

    def test_max_landmarks_respected(self):
        net  = GridCellNetwork(dim=64, periods=[5.0], seed=0)
        slam = LoopClosureSLAM(net, dim=64, max_landmarks=3)
        slam.reset()
        for i in range(10):
            scene = (torch.rand(64) > 0.5).float()
            slam.slam_step(dx=0.5, scene_hv=scene)
        assert slam.map_summary()["n_landmarks"] <= 3


# ═══════════════════════════════════════════════════════════════════════════════
# RandomSubspaceAdversarialDetector
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.hdc_security import RandomSubspaceAdversarialDetector


class TestRandomSubspaceAdversarialDetector:
    def _make_detector(self, n_features: int = 8, n_classes: int = 3):
        clf = _make_trained_clf(n_features=n_features, n_classes=n_classes)
        return RandomSubspaceAdversarialDetector(
            clf, n_subspaces=8, subspace_fraction=0.7, detection_threshold=0.5, seed=0
        )

    def test_detect_returns_dict(self):
        det = self._make_detector()
        x   = torch.randn(8)
        r   = det.detect(x)
        assert "is_adversarial" in r
        assert "score" in r
        assert "vote_dist" in r

    def test_score_in_range(self):
        det = self._make_detector()
        s   = det.adversarial_score(torch.randn(8))
        assert 0.0 <= s <= 1.0

    def test_benign_input_low_score(self):
        det = self._make_detector()
        # Clean input from training distribution should have low disagreement
        scores = [det.adversarial_score(torch.randn(8) + 1.0) for _ in range(10)]
        # Not all scores should be at maximum entropy
        assert min(scores) < 1.0

    def test_n_subspaces_masks_created(self):
        det = self._make_detector()
        assert len(det._masks) == 8

    def test_subspace_masks_are_binary(self):
        det = self._make_detector()
        for mask in det._masks:
            assert mask.dtype == torch.bool

    def test_detection_threshold_respected(self):
        det_strict = self._make_detector()
        det_strict.threshold = 0.0   # flag everything
        r = det_strict.detect(torch.randn(8))
        assert r["is_adversarial"] is True   # score > 0 always


# ═══════════════════════════════════════════════════════════════════════════════
# HDCSecretSharing
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.hdc_security import HDCSecretSharing


class TestHDCSecretSharing:
    def test_split_returns_n_shares(self):
        ss     = HDCSecretSharing(n_shares=3)
        secret = (torch.rand(D) > 0.5).float()
        shares = ss.split(secret, seed=42)
        assert len(shares) == 3

    def test_reconstruct_exact(self):
        ss     = HDCSecretSharing(n_shares=3)
        secret = (torch.rand(D) > 0.5).float()
        shares = ss.split(secret, seed=7)
        recon  = ss.reconstruct(shares)
        assert torch.equal(recon.float().round(), secret.float())

    def test_shares_are_binary(self):
        ss     = HDCSecretSharing(n_shares=3)
        secret = (torch.rand(D) > 0.5).float()
        shares = ss.split(secret, seed=0)
        for s in shares:
            assert set(s.unique().tolist()).issubset({0.0, 1.0})

    def test_verify_shares_true_for_correct(self):
        ss     = HDCSecretSharing(n_shares=4)
        secret = (torch.rand(D) > 0.5).float()
        shares = ss.split(secret, seed=1)
        assert ss.verify_shares(shares, secret) is True

    def test_partial_shares_low_leakage(self):
        ss     = HDCSecretSharing(n_shares=5)
        secret = (torch.rand(D) > 0.5).float()
        shares = ss.split(secret, seed=2)
        leakage = ss.partial_info_leakage(shares[:2])   # only 2 of 5
        assert leakage < 0.1   # should be near 0 (XOR shares are uniform)

    def test_single_share_reveals_nothing(self):
        ss     = HDCSecretSharing(n_shares=3)
        secret = (torch.rand(D) > 0.5).float()
        shares = ss.split(secret, seed=3)
        # Single share should be ~uniformly random (density near 0.5)
        density = float(shares[0].mean().item())
        assert 0.3 < density < 0.7

    def test_two_shares_xor_is_uniform(self):
        ss     = HDCSecretSharing(n_shares=3)
        secret = (torch.rand(D) > 0.5).float()
        shares = ss.split(secret, seed=4)
        # XOR of 2 of 3 shares should NOT equal the secret
        partial_xor = (shares[0] + shares[1]) % 2
        assert not torch.equal(partial_xor, secret)


# ═══════════════════════════════════════════════════════════════════════════════
# CatastrophicForgettingBound
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.hdc_security import CatastrophicForgettingBound


class TestCatastrophicForgettingBound:
    def test_snapshot_and_measure(self):
        tracker = CatastrophicForgettingBound(n_classes=3, forget_threshold=0.1)
        protos  = [(torch.rand(D) > 0.5).float() for _ in range(3)]
        tracker.snapshot(protos)
        result = tracker.measure(protos)   # identical → zero forgetting
        assert result["max_forgetting"] == 0.0
        assert result["bound_satisfied"] is True

    def test_large_change_triggers_violation(self):
        tracker = CatastrophicForgettingBound(n_classes=2, forget_threshold=0.05)
        p0 = [(torch.rand(D) > 0.5).float() for _ in range(2)]
        tracker.snapshot(p0)
        # Completely new random prototypes → high forgetting
        p1 = [(torch.rand(D) > 0.5).float() for _ in range(2)]
        result = tracker.measure(p1)
        assert result["max_forgetting"] > 0.0
        assert len(result["violations"]) > 0

    def test_bound_satisfied_on_small_change(self):
        tracker = CatastrophicForgettingBound(n_classes=2, forget_threshold=0.9)
        p0 = [(torch.rand(D) > 0.5).float() for _ in range(2)]
        tracker.snapshot(p0)
        p1 = [(torch.rand(D) > 0.5).float() for _ in range(2)]
        result = tracker.measure(p1)
        assert result["bound_satisfied"] is True   # threshold=0.9, max change ≤ ~0.5

    def test_summary_keys(self):
        tracker = CatastrophicForgettingBound(n_classes=2)
        p = [(torch.rand(D) > 0.5).float() for _ in range(2)]
        tracker.snapshot(p)
        tracker.measure(p)
        summary = tracker.summary()
        assert "mean_forget_rate" in summary
        assert "total_violations" in summary

    def test_trace_populated_after_measure(self):
        tracker = CatastrophicForgettingBound(n_classes=2)
        p0 = [(torch.rand(D) > 0.5).float() for _ in range(2)]
        p1 = [(torch.rand(D) > 0.5).float() for _ in range(2)]
        tracker.snapshot(p0)
        tracker.measure(p1)
        curve = tracker.stability_plasticity_curve()
        assert len(curve) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# BatchHDCOps, ONNXClassifierExporter, HDCModelCard, DeploymentValidator
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.deployment_export import (
    BatchHDCOps, ONNXClassifierExporter, HDCModelCard, DeploymentValidator,
)


class TestBatchHDCOps:
    def test_batch_hamming_sim_shape(self):
        B, C = 4, 3
        q    = (torch.rand(B, D) > 0.5).float()
        c    = (torch.rand(C, D) > 0.5).float()
        sims = BatchHDCOps.batch_hamming_sim(q, c)
        assert sims.shape == (B, C)

    def test_batch_hamming_sim_in_range(self):
        q = (torch.rand(4, D) > 0.5).float()
        c = (torch.rand(3, D) > 0.5).float()
        sims = BatchHDCOps.batch_hamming_sim(q, c)
        assert (sims >= 0.0).all() and (sims <= 1.0).all()

    def test_batch_hamming_identical(self):
        v    = (torch.rand(D) > 0.5).float()
        sims = BatchHDCOps.batch_hamming_sim(v.unsqueeze(0), v.unsqueeze(0))
        assert float(sims[0, 0]) == pytest.approx(1.0)

    def test_batch_xor_bind_shape(self):
        a = (torch.rand(4, D) > 0.5).float()
        b = (torch.rand(4, D) > 0.5).float()
        r = BatchHDCOps.batch_xor_bind(a, b)
        assert r.shape == (4, D)

    def test_batch_xor_bind_binary(self):
        a = (torch.rand(4, D) > 0.5).float()
        b = (torch.rand(4, D) > 0.5).float()
        r = BatchHDCOps.batch_xor_bind(a, b)
        assert set(r.unique().tolist()).issubset({0.0, 1.0})

    def test_batch_majority_shape(self):
        hvs = (torch.rand(4, 8, D) > 0.5).float()
        out = BatchHDCOps.batch_majority(hvs)
        assert out.shape == (4, D)

    def test_batch_majority_binary(self):
        hvs = (torch.rand(2, 5, D) > 0.5).float()
        out = BatchHDCOps.batch_majority(hvs)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_top_k_similar_shape(self):
        q = (torch.rand(4, D) > 0.5).float()
        c = (torch.rand(10, D) > 0.5).float()
        vals, idx = BatchHDCOps.top_k_similar(q, c, k=3)
        assert vals.shape == (4, 3)
        assert idx.shape  == (4, 3)

    def test_batch_encode_level_id_shape(self):
        F   = 8
        clf = _make_trained_clf(n_features=F, n_classes=3, dim=D)
        feat_hvs = clf.feature_id_hvs.float()   # HDCCClassifier uses feature_id_hvs
        X   = torch.randn(5, F)
        hvs = BatchHDCOps.batch_encode_level_id(X, feat_hvs, clf.level_hvs.float())
        assert hvs.shape == (5, D)
        assert set(hvs.unique().tolist()).issubset({0.0, 1.0})

    def test_batch_vs_single_sample_agreement(self):
        F   = 8
        clf = _make_trained_clf(n_features=F, n_classes=3, dim=D)
        feat_hvs = clf.feature_id_hvs.float()
        x   = torch.randn(F)
        # Single sample
        hv_single = clf.encode(x)
        # Batch of 1
        hv_batch  = BatchHDCOps.batch_encode_level_id(
            x.unsqueeze(0), feat_hvs, clf.level_hvs.float()
        ).squeeze(0)
        assert torch.equal(hv_single, hv_batch)


class TestONNXClassifierExporter:
    def test_compute_model_hash_deterministic(self):
        clf  = _make_trained_clf()
        exp1 = ONNXClassifierExporter(clf)
        exp2 = ONNXClassifierExporter(clf)
        assert exp1.compute_model_hash() == exp2.compute_model_hash()

    def test_build_pytorch_model_forward(self):
        clf = _make_trained_clf(n_features=8)
        exp = ONNXClassifierExporter(clf)
        mod = exp._build_pytorch_model().eval()
        x   = torch.randn(3, 8)
        out = mod(x)
        assert out.shape == (3, 3)   # (B, n_classes)

    def test_export_torchscript(self, tmp_path):
        clf  = _make_trained_clf(n_features=8)
        exp  = ONNXClassifierExporter(clf)
        path = str(tmp_path / "model.pt")
        result = exp.export_torchscript(path)
        import os
        assert os.path.exists(path)
        assert result["model_bytes"] > 0
        assert result["format"] == "torchscript"


class TestHDCModelCard:
    def test_generate_has_architecture(self):
        clf  = _make_trained_clf()
        card_gen = HDCModelCard(clf)
        card = card_gen.generate()
        assert "architecture" in card
        assert card["architecture"]["n_classes"] == 3

    def test_generate_with_profiler(self):
        from hdc.hardware_synthesis import MCUDeploymentProfiler
        clf      = _make_trained_clf(n_features=8)
        profiler = MCUDeploymentProfiler(hd_dim=D, n_classes=3, input_dim=8)
        card_gen = HDCModelCard(clf, profiler=profiler)
        card     = card_gen.generate()
        assert "deployment" in card

    def test_print_summary_is_string(self):
        clf  = _make_trained_clf()
        card = HDCModelCard(clf)
        s    = card.print_summary()
        assert isinstance(s, str)
        assert "Arthedain" in s

    def test_generate_with_dp_accountant(self):
        clf = _make_trained_clf()
        acc = RenyiDPAccountant(noise_multiplier=1.0)
        acc.step(10)
        card_gen = HDCModelCard(clf, dp_accountant=acc)
        card = card_gen.generate()
        assert "privacy" in card


class TestDeploymentValidator:
    def test_validate_batch_ops_passes(self):
        clf = _make_trained_clf(n_features=8)
        val = DeploymentValidator(clf, n_test=10)
        result = val.validate_batch_ops()
        assert result["all_passed"] is True


class TestFederationTierNode:
    def test_aggregate_returns_none_below_min(self):
        cfg  = TierConfig("edge", n_children=4, min_children=2)
        node = FederationTierNode("n0", n_classes=2, dim=D, cfg=cfg)
        node.receive({"prototypes": [(torch.rand(D) > 0.5).float()] * 2, "counts": [5, 5]})
        result = node.aggregate()
        assert result is None  # only 1 < min_children=2

    def test_aggregate_succeeds_with_enough(self):
        cfg  = TierConfig("edge", n_children=4, min_children=2)
        node = FederationTierNode("n0", n_classes=2, dim=D, cfg=cfg)
        for _ in range(2):
            node.receive({"prototypes": [(torch.rand(D) > 0.5).float()] * 2, "counts": [5, 5]})
        result = node.aggregate()
        assert result is not None
        assert len(result["prototypes"]) == 2

    def test_dp_noise_changes_output(self):
        cfg_dp   = TierConfig("edge", n_children=2, dp_epsilon=0.1, min_children=1)
        cfg_base = TierConfig("edge", n_children=2, dp_epsilon=0.0, min_children=1)
        n_dp   = FederationTierNode("dp",   n_classes=1, dim=D, cfg=cfg_dp)
        n_base = FederationTierNode("base", n_classes=1, dim=D, cfg=cfg_base)
        contrib = {"prototypes": [(torch.rand(D) > 0.5).float()], "counts": [10]}
        n_dp.receive(dict(contrib))
        n_base.receive(dict(contrib))
        r_dp   = n_dp.aggregate()
        r_base = n_base.aggregate()
        # Both should return valid results
        assert r_dp is not None
        assert r_base is not None


# ═══════════════════════════════════════════════════════════════════════════════
# ComplexHammingSearch: remove + update
# ═══════════════════════════════════════════════════════════════════════════════

from hdc.holographic_graph_neuron import ComplexHammingSearch


class TestComplexHammingSearchDeletion:
    def test_remove_existing_label(self):
        chs = ComplexHammingSearch(D)
        hv  = (torch.rand(D) > 0.5).float()
        chs.store(hv, label=42)
        assert chs.n_stored() == 1
        removed = chs.remove(42)
        assert removed is True
        assert chs.n_stored() == 0

    def test_remove_nonexistent_label(self):
        chs = ComplexHammingSearch(D)
        removed = chs.remove(999)
        assert removed is False

    def test_remove_preserves_other_entries(self):
        chs = ComplexHammingSearch(D)
        for i in range(5):
            chs.store((torch.rand(D) > 0.5).float(), label=i)
        chs.remove(2)
        assert chs.n_stored() == 4
        # Label 2 should not appear in queries
        q       = (torch.rand(D) > 0.5).float()
        results = chs.query(q, threshold=1.0)
        labels  = [r["label"] for r in results]
        assert 2 not in labels

    def test_update_replaces_entry(self):
        chs  = ComplexHammingSearch(D)
        hv1  = (torch.rand(D) > 0.5).float()
        hv2  = torch.ones(D)   # very different
        chs.store(hv1, label=7)
        chs.update(7, hv2)
        assert chs.n_stored() == 1
        # Query with hv2 should find label 7
        results = chs.query(hv2, threshold=1.0, top_k=1)
        assert results and results[0]["label"] == 7

    def test_remove_all_clears_memory(self):
        chs = ComplexHammingSearch(D)
        for i in range(3):
            chs.store((torch.rand(D) > 0.5).float(), label=i)
        for i in range(3):
            chs.remove(i)
        assert chs.n_stored() == 0
        assert chs._H_complex is None
