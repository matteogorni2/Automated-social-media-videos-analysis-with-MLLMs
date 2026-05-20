## 🔍 Overview

This repository contains the complete codebase for a master's thesis project focused on **automated forensic annotation of social media videos** and the **creation of textual video descriptions** using the **Qwen3-VL-32B-Instruct** multimodal large language model.

### Goals
- Develop a scalable pipeline for analyzing sexualized content in TikTok videos
- Replace/augment manual human coding with MLLM-based structured annotation
- Enable quantitative analysis of algorithmic curation effects on self-objectification

### Key Features
✅ **Offline-capable inference** for restricted/HPC environments  
✅ **Multi-GPU parallelization** via Accelerate + SLURM  
✅ **Structured JSON output** 
✅ **Singularity containerization** for reproducible dependencies  
✅ **Modular orchestration**: manifest-driven job submission  
✅ **Post-hoc analysis notebook** for statistical exploration of results  

## Repository Structure

```text
Automated-social-media-videos-analysis-with-MLLMs/
├── thesis_code/
│   ├── qwen_inference.py
│   │   └── Core inference pipeline: video → JSON annotations
│   │
│   ├── orchestrator_script.py
│   │   └── HPC job submission wrapper
│   │
│   ├── job_template.sbatch
│   │   └── SLURM job template
│   │
│   ├── video_captions_manifest.yml
│   │   └── Configuration for video caption generation
│   │
│   └── manifest.yml
│       └── Central configuration file
│
├── results_analysis_and_plots.ipynb
│   ├── Loads annotation JSON files and metadata
│   ├── Computes derived metrics (e.g., sexualization score)
│   └── Performs statistical analysis and visualization
│       ├── ANOVA
│       ├── Tukey HSD
│       └── Plots and charts
│
├── dissertation.pdf
│   └── Full thesis document
│
└── README.md
    └── Project documentation
```
