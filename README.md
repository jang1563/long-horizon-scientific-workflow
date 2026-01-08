# Long-Horizon Scientific Workflow Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for executing multi-stage scientific discovery workflows with explicit reasoning traces, checkpoint validation, and ground truth evaluation. Designed to demonstrate AI capabilities for complex, long-horizon reasoning tasks in life sciences.

> **Note**: This public repository contains the framework architecture and example templates. A full demonstration with real spaceflight mission data (including unpublished results) is available upon request for interview/evaluation purposes. Please contact the author for access.

## 🎯 Overview

This project implements an **8-stage scientific discovery pipeline** for analyzing spaceflight biomarker data from cfRNA-seq experiments. It demonstrates key capabilities relevant to AI systems in scientific research:

- **Structured Reasoning**: Explicit reasoning traces with confidence scores
- **Long-Horizon Planning**: Multi-stage pipeline with dependencies
- **Verifiable Outputs**: Ground truth comparison and checkpoint validation
- **Domain Translation**: Statistical findings → Biological interpretation

## 🔬 Scientific Context

The workflow analyzes circulating cell-free RNA sequencing (cfRNA-seq) data to identify reproducible molecular biomarkers of spaceflight stress. The framework is designed to detect signatures such as:

- **Erythroid Suppression**: Cross-mission concordance analysis
- **Transient Response**: Recovery timepoint comparison
- **Space Anemia Signature**: Molecular-level validation through cross-study comparison

## 📊 Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LONG-HORIZON SCIENTIFIC WORKFLOW                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STAGE 1: Data Ingestion & QC ──────────────────────────────────►   │
│                           ↓                                         │
│  STAGE 2: Exploratory Analysis ─────────────────────────────────►   │
│                           ↓                                         │
│  STAGE 3: Statistical Analysis ─────────────────────────────────►   │
│                           ↓                                         │
│  STAGE 4: Cross-Study Validation ───────────────────────────────►   │
│                           ↓                                         │
│  STAGE 5: Biological Interpretation ────────────────────────────►   │
│                           ↓                                         │
│  STAGE 6: Hypothesis Generation ────────────────────────────────►   │
│                           ↓                                         │
│  STAGE 7: Experimental Design ──────────────────────────────────►   │
│                           ↓                                         │
│  STAGE 8: Scientific Communication ─────────────────────────────►   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/jang1563/long-horizon-scientific-workflow.git
cd long-horizon-scientific-workflow
pip install -r requirements.txt
```

### Running the Workflow

```bash
# Execute the full workflow (requires input data)
python src/workflow_engine.py

# Generate visualizations (requires workflow outputs)
python src/visualize_results.py
```

### Using Your Own Data

```python
from src.workflow_engine import WorkflowEngine

# Initialize with custom specification
engine = WorkflowEngine('path/to/your/workflow_spec.json')

# Execute workflow
report = engine.run_workflow()

# Access results
print(f"Completion Rate: {report['evaluation_metrics']['workflow_completion_rate']:.0%}")
print(f"Scientific Accuracy: {report['evaluation_metrics']['scientific_accuracy']:.0%}")
```

## 📁 Project Structure

```
long-horizon-scientific-workflow/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── LICENSE                       # MIT License
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── workflow_engine.py        # Main execution engine
│   ├── workflow_spec.json        # Workflow specification
│   └── visualize_results.py      # Visualization generation
│
├── docs/                         # Documentation
│   ├── WORKFLOW_SPECIFICATION.md # Detailed spec documentation
│   ├── REASONING_TRACE.md        # Example reasoning traces
│   └── EVALUATION_REPORT.md      # Evaluation report template
│
├── outputs/                      # Generated outputs (empty in public repo)
│   ├── figures/                  # Visualizations
│   └── reports/                  # Generated reports
│
└── tests/                        # Unit tests
    └── test_workflow.py
```

## 🧠 Reasoning Trace System

The framework captures detailed reasoning traces at each step:

```python
{
    "timestamp": "2026-01-08T19:41:01.456789",
    "stage_id": "stage_3_statistical_analysis",
    "reasoning_type": "decision",
    "content": "Selected relaxed threshold: padj<0.10, |LFC|>0.5",
    "confidence": 0.85,
    "evidence": [
        "Sample size is small (n=4)",
        "Standard padj<0.05 may be too stringent",
        "Relaxed threshold balances discovery vs false positives"
    ],
    "alternatives_considered": [
        "Stringent (padj<0.05, |LFC|>1.0): Too few DEGs expected",
        "Standard (padj<0.05, |LFC|>0.5): May miss relevant genes",
        "Exploratory (padj<0.20): Too many false positives"
    ]
}
```

### Reasoning Types

| Type | Description | Example |
|------|-------------|---------|
| `observation` | Data-driven findings | "Loaded 5,346 genes from dataset" |
| `inference` | Logical conclusions | "Low correlation expected due to noise" |
| `decision` | Choice points | "Selected relaxed threshold" |
| `uncertainty` | Acknowledged limitations | "Alternative explanations must be considered" |
| `verification` | Validation checkpoints | "Schema validation passed" |

## 📈 Evaluation Metrics

The framework evaluates workflow execution across multiple dimensions:

| Metric | Description | Target |
|--------|-------------|--------|
| Workflow Completion Rate | Stages successfully completed | 100% |
| Scientific Accuracy | Ground truth match rate | 90% |
| Time Efficiency | Actual vs estimated duration | 1.0x |
| Average Stage Score | Mean checkpoint score | 0.70 |

## 🔧 Customization

### Creating Custom Workflow Specifications

```json
{
  "workflow_metadata": {
    "name": "Your Workflow Name",
    "version": "1.0.0",
    "domain": "Life Sciences"
  },
  "stages": [
    {
      "id": "stage_1",
      "name": "Data Ingestion",
      "order": 1,
      "success_criteria": {
        "files_loaded": {"type": "boolean", "expected": true}
      },
      "reasoning_prompts": [
        "What is the structure of the data?",
        "Are there quality issues?"
      ]
    }
  ],
  "evaluation_framework": {
    "ground_truth_comparisons": [
      {"metric": "expected_result", "expected": 100}
    ]
  }
}
```

### Adding Custom Stage Handlers

```python
class CustomWorkflowEngine(WorkflowEngine):
    def _execute_stage_custom(self, spec):
        reasoning = []

        # Log observation
        reasoning.append(self.log_reasoning(
            spec['id'],
            ReasoningType.OBSERVATION,
            "Starting custom analysis",
            confidence=1.0
        ))

        # Your analysis code here
        results = perform_analysis()

        # Log decision
        reasoning.append(self.log_reasoning(
            spec['id'],
            ReasoningType.DECISION,
            f"Selected method X based on {criteria}",
            confidence=0.85,
            alternatives=["Method Y", "Method Z"]
        ))

        return outputs, metrics, reasoning
```

## 🎯 Relevance to AI Safety & Capabilities

This framework demonstrates key principles for beneficial AI systems:

### 1. Reliability
- Checkpoint-based validation at each stage
- Explicit success criteria with measurable thresholds
- Error handling and recovery mechanisms

### 2. Interpretability
- Full reasoning traces with timestamps
- Confidence scores for each inference
- Evidence and alternatives documented

### 3. Steerability
- Modular stage architecture
- Configurable thresholds and parameters
- Human review checkpoints for critical decisions

### 4. Domain Expertise Integration
- Biology-informed decision points
- Literature-grounded interpretations
- Testable hypothesis generation

## 📚 References

- **Spaceflight Biology**: NASA GeneLab, Space Omics datasets
- **cfRNA Analysis**: DESeq2, Tissue deconvolution methods
- **ML for Biology**: Cross-study validation, Biomarker discovery

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**JangKeun Kim** - [jang1563](https://github.com/jang1563)

## 🙏 Acknowledgments

- Commercial spaceflight mission teams
- Anthropic for inspiring this framework design
- Open-source bioinformatics community

---

<p align="center">
  <i>Designed to demonstrate AI capabilities for complex, long-horizon scientific reasoning</i>
</p>
