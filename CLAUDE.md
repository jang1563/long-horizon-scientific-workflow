# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a framework for executing multi-stage scientific discovery workflows with explicit reasoning traces, checkpoint validation, and ground truth evaluation. It demonstrates AI capabilities for complex, long-horizon reasoning tasks in life sciences, specifically analyzing spaceflight biomarker data from cfRNA-seq experiments.

## Commands

### Run the Workflow
```bash
python src/workflow_engine.py
```
Note: The workflow expects data files in the configured data directory and a workflow spec at `/home/claude/workflow_spec.json`. For local development, update the path in `workflow_engine.py:1222` or create a custom spec.

### Generate Visualizations
```bash
python src/visualize_results.py
```
Requires workflow outputs to exist in the configured output directory.

### Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_workflow.py -v

# Run a specific test class
pytest tests/test_workflow.py::TestCheckpointEvaluation -v

# Run with coverage
pytest tests/ --cov=src
```

### Install Dependencies
```bash
pip install -r requirements.txt

# Or install as package with dev dependencies
pip install -e ".[dev]"
```

## Architecture

### Core Components

**WorkflowEngine** (`src/workflow_engine.py`) - The main execution engine that:
- Loads workflow specifications from JSON
- Executes 8 sequential stages with dependencies
- Captures reasoning traces at each step
- Evaluates checkpoints against success criteria
- Generates execution reports with ground truth comparison

**Stage Handlers** - Each stage is implemented as a method `_execute_stage_N_*` that returns a tuple of `(outputs, metrics, reasoning)`. The handler is dynamically dispatched based on the stage ID.

**Data Structures**:
- `ReasoningEntry` - Captures reasoning with timestamp, type, content, confidence, evidence, and alternatives
- `CheckpointResult` - Tracks criterion evaluation against expected values
- `StageResult` - Complete stage execution result including outputs, metrics, and reasoning trace
- `ReasoningType` enum: observation, inference, decision, uncertainty, verification, error_recovery
- `StageStatus` enum: pending, running, completed, failed, skipped

### Workflow Specification (`src/workflow_spec.json`)

JSON-driven configuration defining:
- Stage definitions with inputs, outputs, success criteria
- Decision points with options and required justifications
- Ground truth values for validation
- Reasoning prompts to guide analysis

### 8-Stage Pipeline

1. **Data Ingestion & QC** - Load datasets, validate schemas
2. **Exploratory Analysis** - Distribution analysis, outlier detection
3. **Statistical Analysis** - DEG identification with threshold selection
4. **Cross-Study Validation** - Correlation between Mission A and Mission B
5. **Biological Interpretation** - Pathway analysis, erythroid signature detection
6. **Hypothesis Generation** - Mechanistic hypotheses with testable predictions
7. **Experimental Design** - Validation study proposals with power analysis
8. **Scientific Communication** - Report generation

### Key Patterns

**Reasoning Logging**: Use `log_reasoning()` to capture decisions:
```python
self.log_reasoning(
    stage_id,
    ReasoningType.DECISION,
    "Selected threshold X because...",
    confidence=0.85,
    evidence=["reason1", "reason2"],
    alternatives=["option A", "option B"]
)
```

**Checkpoint Evaluation**: Success criteria are evaluated automatically based on type (boolean, threshold) and weight.

**Shared Context**: Stages share data via `self.shared_context` dictionary, carrying forward datasets, results, and intermediate artifacts.

## Extending the Framework

To add a new stage handler:
1. Add stage specification to `workflow_spec.json`
2. Create method `_execute_stage_N_name(self, spec)` returning `(outputs, metrics, reasoning)`
3. Store results in `self.shared_context` for downstream stages
