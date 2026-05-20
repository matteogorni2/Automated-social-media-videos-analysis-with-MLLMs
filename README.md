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

### Repo Structure 
'''text
Automated-social-media-videos-analysis-with-MLLMs/
├── thesis_code/
│   ├── qwen_inference.py        # Core inference: video → JSON annotation
│   ├── orchestrator_script.py   # HPC job submission wrapper
│   ├── job_template.sbatch      # SLURM template
│   ├── video_captions_manifest.yml # Configuration file for the creation of video descriptions
│   └── manifest.yml             # Central configuration file
├── results_analysis_and_plots.ipynb  # Post-processing notebook
│   • Loads annotation JSON + metadata Excel
│   • Computes derived metrics (e.g., sexualization score)
│   • Statistical testing (ANOVA, Tukey HSD) & visualization
│
├── dissertation.pdf             # Full thesis document
│
└── README.md                    # This file
'''
