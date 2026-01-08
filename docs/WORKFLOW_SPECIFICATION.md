# Workflow Specification Documentation

## Overview

The workflow specification defines the structure, stages, and evaluation criteria for the long-horizon scientific workflow. This document provides detailed documentation for each component.

## Specification Schema

### Metadata

```json
{
  "workflow_metadata": {
    "name": "Spaceflight Biomarker Discovery Pipeline",
    "version": "1.0.0",
    "description": "Long-horizon scientific workflow for identifying reproducible molecular biomarkers",
    "domain": "Life Sciences / Spaceflight Biology",
    "estimated_duration_minutes": 30,
    "complexity_level": "advanced"
  }
}
```

### Global Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_directory` | string | Path to input data files |
| `output_directory` | string | Path for generated outputs |
| `log_level` | string | Logging verbosity (minimal, standard, detailed) |
| `checkpoint_on_failure` | boolean | Save state on stage failure |
| `max_retries_per_stage` | integer | Maximum retry attempts |
| `reasoning_trace_enabled` | boolean | Enable reasoning logging |

---

## Stage Definitions

### Stage 1: Data Ingestion & Quality Control

**Purpose**: Load and validate input datasets

**Inputs**:
- DESeq2 differential expression results (CSV format)
- Expected columns: gene_name, log2FoldChange, pvalue, padj, baseMean

**Outputs**:
- `data_inventory.json`: File metadata
- `qc_report.json`: Quality metrics
- `validated_datasets.pkl`: Cleaned data

**Success Criteria**:
| Criterion | Type | Threshold | Weight |
|-----------|------|-----------|--------|
| all_files_loaded | boolean | true | 1.0 |
| schema_valid | boolean | true | 1.0 |
| missing_rate | threshold | max 0.10 | 0.8 |
| min_genes_per_dataset | threshold | min 1000 | 0.7 |

**Reasoning Prompts**:
1. What is the structure of each dataset?
2. Are there any data quality issues?
3. Are datasets compatible for comparison?

---

### Stage 2: Exploratory Data Analysis

**Purpose**: Understand data distributions and identify patterns

**Outputs**:
- `eda_summary.json`: Distribution statistics
- `distribution_plots.png`: Visualizations
- `outlier_report.json`: Anomaly detection

**Success Criteria**:
| Criterion | Type | Threshold | Weight |
|-----------|------|-----------|--------|
| distributions_analyzed | boolean | true | 1.0 |
| outliers_identified | boolean | true | 0.8 |
| cross_dataset_overlap_pct | threshold | min 0.5 | 0.9 |

---

### Stage 3: Statistical Analysis

**Purpose**: Identify differentially expressed genes

**Decision Points**:
- **Threshold Selection**: Choose significance cutoffs
  - Options: stringent_0.05, standard_0.05_lfc0.5, relaxed_0.10, exploratory_0.20
  - Justification required

**Outputs**:
- `deg_results.json`: DEG lists per condition
- `threshold_justification.md`: Decision rationale
- `volcano_plots.png`: Visualization

**Success Criteria**:
| Criterion | Type | Threshold | Weight |
|-----------|------|-----------|--------|
| threshold_justified | boolean | true | 1.0 |
| degs_identified | threshold | min 1 | 0.9 |
| multiple_testing_corrected | boolean | true | 1.0 |

---

### Stage 4: Cross-Study Validation

**Purpose**: Validate findings across independent datasets

**Outputs**:
- `cross_study_results.json`: Correlation metrics
- `correlation_plots.png`: Scatter plots
- `conserved_genes.csv`: Reproducible gene list

**Success Criteria**:
| Criterion | Type | Threshold | Weight |
|-----------|------|-----------|--------|
| correlation_computed | boolean | true | 1.0 |
| concordant_genes_found | threshold | min 10 | 0.9 |
| reproducibility_assessed | boolean | true | 1.0 |

---

### Stage 5: Biological Interpretation

**Purpose**: Interpret findings in biological context

**Knowledge Sources**:
- Gene Ontology
- KEGG Pathways
- Tissue-specific signatures
- Domain literature

**Outputs**:
- `pathway_enrichment.json`: Enrichment results
- `tissue_deconvolution.json`: Tissue analysis
- `biological_narrative.md`: Interpretation document

**Success Criteria**:
| Criterion | Type | Threshold | Weight |
|-----------|------|-----------|--------|
| pathways_analyzed | boolean | true | 1.0 |
| tissue_origins_identified | boolean | true | 0.9 |
| narrative_coherent | boolean | true | 0.8 |

---

### Stage 6: Hypothesis Generation

**Purpose**: Generate testable mechanistic hypotheses

**Requires Human Review**: Yes

**Outputs**:
- `hypotheses.json`: Structured hypothesis list
- `predictions.json`: Testable predictions
- `alternative_explanations.md`: Alternative interpretations

**Success Criteria**:
| Criterion | Type | Threshold | Weight |
|-----------|------|-----------|--------|
| hypotheses_generated | threshold | min 3 | 1.0 |
| hypotheses_testable | boolean | true | 1.0 |
| alternatives_considered | boolean | true | 0.8 |

---

### Stage 7: Experimental Design

**Purpose**: Design validation experiments

**Requires Human Review**: Yes

**Outputs**:
- `experimental_protocol.md`: Detailed protocols
- `power_analysis.json`: Sample size calculations
- `expected_outcomes.json`: Predictions

**Success Criteria**:
| Criterion | Type | Threshold | Weight |
|-----------|------|-----------|--------|
| experiments_defined | threshold | min 2 | 1.0 |
| power_analysis_done | boolean | true | 0.9 |
| controls_specified | boolean | true | 1.0 |

---

### Stage 8: Scientific Communication

**Purpose**: Generate publication-ready outputs

**Outputs**:
- `final_report.docx`: Comprehensive report
- `presentation.pptx`: Slide deck
- `figures/`: All visualizations
- `supplementary_data.xlsx`: Data tables

**Success Criteria**:
| Criterion | Type | Threshold | Weight |
|-----------|------|-----------|--------|
| report_generated | boolean | true | 1.0 |
| figures_embedded | boolean | true | 0.9 |
| methods_documented | boolean | true | 1.0 |

---

## Evaluation Framework

### Stage Completion Scoring

```python
score = sum(criterion.score for criterion in checkpoints) / sum(criterion.weight for criterion in checkpoints)
passing = score >= 0.7
```

### Overall Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| workflow_completion_rate | Percentage of stages completed | 1.0 |
| reasoning_quality_score | Quality of reasoning traces | 0.8 |
| scientific_accuracy | Match with ground truth | 0.9 |
| time_efficiency | Actual vs estimated duration | 1.0 |

### Ground Truth Comparisons

Pre-defined expected values for validation (configurable per dataset):
- Mission A T1 DEG count
- Mission A Recovery DEG count
- Common genes between missions
- Erythroid concordance
- Conserved genes count

---

## Reasoning Trace Schema

```json
{
  "entry_format": {
    "timestamp": "ISO8601",
    "stage_id": "string",
    "reasoning_type": "enum[observation, inference, decision, uncertainty, verification]",
    "content": "string",
    "confidence": "float[0-1]",
    "evidence": "array[string]",
    "alternatives_considered": "array[string]"
  }
}
```

### Reasoning Types

| Type | When to Use |
|------|-------------|
| observation | Recording data-driven findings |
| inference | Drawing logical conclusions |
| decision | Making choices between alternatives |
| uncertainty | Acknowledging limitations |
| verification | Confirming correctness |
| error_recovery | Handling failures |

---

## Extending the Specification

### Adding New Stages

1. Define stage in `stages` array
2. Implement handler method: `_execute_stage_{id}`
3. Define success criteria
4. Add reasoning prompts

### Custom Success Criteria Types

- `boolean`: Exact match (true/false)
- `threshold`: Numeric comparison (min/max)
- `range`: Value within bounds
- `pattern`: Regex matching

### Custom Reasoning Types

Extend `ReasoningType` enum:

```python
class ReasoningType(Enum):
    OBSERVATION = "observation"
    INFERENCE = "inference"
    DECISION = "decision"
    UNCERTAINTY = "uncertainty"
    VERIFICATION = "verification"
    ERROR_RECOVERY = "error_recovery"
    CUSTOM_TYPE = "custom_type"  # Add your type
```
