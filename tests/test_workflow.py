#!/usr/bin/env python3
"""
Unit tests for Long-Horizon Scientific Workflow Framework
"""

import pytest
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from workflow_engine import (
    WorkflowEngine, 
    ReasoningType, 
    StageStatus,
    ReasoningEntry,
    CheckpointResult
)


class TestReasoningEntry:
    """Tests for ReasoningEntry dataclass"""
    
    def test_creation(self):
        entry = ReasoningEntry(
            timestamp="2026-01-08T12:00:00",
            stage_id="test_stage",
            reasoning_type="observation",
            content="Test content",
            confidence=0.9,
            evidence=["Evidence 1"],
            alternatives_considered=["Alt 1"]
        )
        assert entry.confidence == 0.9
        assert entry.stage_id == "test_stage"
    
    def test_to_dict(self):
        entry = ReasoningEntry(
            timestamp="2026-01-08T12:00:00",
            stage_id="test_stage",
            reasoning_type="observation",
            content="Test content",
            confidence=0.9
        )
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d['confidence'] == 0.9


class TestCheckpointResult:
    """Tests for CheckpointResult dataclass"""
    
    def test_passing_checkpoint(self):
        result = CheckpointResult(
            criterion="test_criterion",
            expected=True,
            actual=True,
            passed=True,
            weight=1.0,
            score=1.0
        )
        assert result.passed == True
        assert result.score == 1.0
    
    def test_failing_checkpoint(self):
        result = CheckpointResult(
            criterion="test_criterion",
            expected=10,
            actual=5,
            passed=False,
            weight=1.0,
            score=0.0
        )
        assert result.passed == False
        assert result.score == 0.0


class TestReasoningType:
    """Tests for ReasoningType enum"""
    
    def test_all_types_exist(self):
        expected_types = [
            'observation', 'inference', 'decision', 
            'uncertainty', 'verification', 'error_recovery'
        ]
        for t in expected_types:
            assert hasattr(ReasoningType, t.upper())
    
    def test_enum_values(self):
        assert ReasoningType.OBSERVATION.value == "observation"
        assert ReasoningType.DECISION.value == "decision"


class TestStageStatus:
    """Tests for StageStatus enum"""
    
    def test_all_statuses_exist(self):
        expected_statuses = ['pending', 'running', 'completed', 'failed', 'skipped']
        for s in expected_statuses:
            assert hasattr(StageStatus, s.upper())


class TestWorkflowSpecification:
    """Tests for workflow specification parsing"""
    
    @pytest.fixture
    def sample_spec(self, tmp_path):
        spec = {
            "workflow_metadata": {
                "name": "Test Workflow",
                "version": "1.0.0"
            },
            "global_config": {
                "data_directory": str(tmp_path),
                "output_directory": str(tmp_path / "outputs"),
                "log_level": "minimal",
                "checkpoint_on_failure": True,
                "max_retries_per_stage": 1,
                "reasoning_trace_enabled": True
            },
            "stages": [],
            "evaluation_framework": {
                "stage_completion_scoring": {
                    "method": "weighted_criteria",
                    "passing_threshold": 0.7
                },
                "ground_truth_comparisons": []
            }
        }
        spec_path = tmp_path / "test_spec.json"
        with open(spec_path, 'w') as f:
            json.dump(spec, f)
        return spec_path
    
    def test_spec_loading(self, sample_spec):
        engine = WorkflowEngine(str(sample_spec))
        assert engine.spec['workflow_metadata']['name'] == "Test Workflow"
    
    def test_output_directory_creation(self, sample_spec):
        engine = WorkflowEngine(str(sample_spec))
        assert os.path.exists(engine.output_dir)


class TestCheckpointEvaluation:
    """Tests for checkpoint evaluation logic"""
    
    @pytest.fixture
    def engine(self, tmp_path):
        spec = {
            "workflow_metadata": {"name": "Test"},
            "global_config": {
                "data_directory": str(tmp_path),
                "output_directory": str(tmp_path / "outputs"),
                "log_level": "minimal",
                "checkpoint_on_failure": True,
                "max_retries_per_stage": 1,
                "reasoning_trace_enabled": True
            },
            "stages": [],
            "evaluation_framework": {"ground_truth_comparisons": []}
        }
        spec_path = tmp_path / "test_spec.json"
        with open(spec_path, 'w') as f:
            json.dump(spec, f)
        return WorkflowEngine(str(spec_path))
    
    def test_boolean_checkpoint_pass(self, engine):
        spec = {"type": "boolean", "expected": True, "weight": 1.0}
        result = engine.evaluate_checkpoint("test", spec, True)
        assert result.passed == True
        assert result.score == 1.0
    
    def test_boolean_checkpoint_fail(self, engine):
        spec = {"type": "boolean", "expected": True, "weight": 1.0}
        result = engine.evaluate_checkpoint("test", spec, False)
        assert result.passed == False
        assert result.score == 0.0
    
    def test_threshold_min_pass(self, engine):
        spec = {"type": "threshold", "min": 10, "weight": 0.8}
        result = engine.evaluate_checkpoint("test", spec, 15)
        assert result.passed == True
        assert result.score == 0.8
    
    def test_threshold_min_fail(self, engine):
        spec = {"type": "threshold", "min": 10, "weight": 0.8}
        result = engine.evaluate_checkpoint("test", spec, 5)
        assert result.passed == False
        assert result.score == 0.0
    
    def test_threshold_max_pass(self, engine):
        spec = {"type": "threshold", "max": 0.1, "weight": 0.5}
        result = engine.evaluate_checkpoint("test", spec, 0.05)
        assert result.passed == True
        assert result.score == 0.5


class TestReasoningLogging:
    """Tests for reasoning trace logging"""
    
    @pytest.fixture
    def engine(self, tmp_path):
        spec = {
            "workflow_metadata": {"name": "Test"},
            "global_config": {
                "data_directory": str(tmp_path),
                "output_directory": str(tmp_path / "outputs"),
                "log_level": "detailed",
                "checkpoint_on_failure": True,
                "max_retries_per_stage": 1,
                "reasoning_trace_enabled": True
            },
            "stages": [],
            "evaluation_framework": {"ground_truth_comparisons": []}
        }
        spec_path = tmp_path / "test_spec.json"
        with open(spec_path, 'w') as f:
            json.dump(spec, f)
        return WorkflowEngine(str(spec_path))
    
    def test_log_reasoning(self, engine):
        entry = engine.log_reasoning(
            "test_stage",
            ReasoningType.OBSERVATION,
            "Test observation",
            confidence=0.9,
            evidence=["Evidence 1", "Evidence 2"]
        )
        assert len(engine.global_reasoning_trace) == 1
        assert entry.content == "Test observation"
        assert entry.confidence == 0.9
    
    def test_multiple_reasoning_entries(self, engine):
        engine.log_reasoning("s1", ReasoningType.OBSERVATION, "Obs 1", 0.9)
        engine.log_reasoning("s1", ReasoningType.INFERENCE, "Inf 1", 0.8)
        engine.log_reasoning("s1", ReasoningType.DECISION, "Dec 1", 0.7)
        assert len(engine.global_reasoning_trace) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
