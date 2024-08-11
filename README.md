# Project Overview

This README provides an overview of the methods used and the data collection process in this project.

## Methods

- **Statistical Methods**  
  Applied to compare the distributions between different record linkage tasks.

- **Graph Clustering**  
  Utilized to cluster the linkage tasks into strongly connected components.

- **Active Learning**  
  Employed to label selected tasks efficiently.

- **Machine Learning**  
  Algorithms are trained on linkage tasks labeled using active learning techniques.

- **Evaluation Metrics**  
  Used to assess the performance of the proposed framework.

## Data Collection

- **Camera Datasets**  
  Data collected from approximately 23 different sources.

   
   
### Data collection: 
*  Camera datasets (There are around 23 data sources)

### Repository structure

``` plain
├── 📁 data                   
│   ├── 📁 raw_data             <-- Unprocessed camera datasets
│   ├── 📁 cleaned_data         <-- Preprocessed camera datasets
│   ├── 📁 help_data            <-- Auxiliary data used during preprocessing
│   ├── 📁 linkage_tasks        <-- Linkage tasks for each pair of sources, with the first two columns containing record IDs
│   ├── 📁 linkage_tasks_labeled<-- Labeled linkage tasks generated using active learning
│   └── 📁 ground_truth_data    <-- Ground truth data for the linkage tasks

├── 📁 results                  
│   ├── 📁 statistical_tests    <-- Contains images of statistical test results
│   ├── 📁 clustering           <-- Contains figures of the generated clusters
│   └── 📁 linkage_results      <-- Precision, Recall, and F1-scores for the linkage tasks

├── 📁 meta_tl                  <-- Project codebase
│   ├── 📃 statistical_tests.py     <-- Implementations of statistical methods for comparing linkage task distributions
│   ├── 📃 graph_clustering.py      <-- Methods for graph clustering of linkage tasks
│   ├── 📃 model_selection.py       <-- Methods for selecting linkage tasks from clusters
│   ├── 📃 preprocessing.py         <-- Data preprocessing scripts
│   ├── 📃 active_learning.py       <-- Active learning methods for labeling selected tasks
│   ├── 📃 evaluation.py            <-- Implementation of evaluation methods for assessing the proposed methods
│   ├── 📁 record_linkage           <-- Implementation of the record linkage process
│   │   ├── 📃 blocking.py          <-- Methods for blocking
│   │   ├── 📃 comparison.py        <-- Methods for comparing records sharing the same blocking keys
│   │   ├── 📃 load_data_set.py     <-- Prepares datasets for the record linkage process
│   │   └── 📃 record_linkage_main.py <-- Main script for record linkage
│   ├── 📃 utils.py                 <-- Helper functions used throughout the project
│   ├── 📁 data_pre_processing      <-- Data preprocessing scripts
│   │   ├── 📃 data_cleaning.py     <-- Various methods used to clean raw data sources
│   │   └── 📃 generate_ground_truth_data.py  <-- Generates ground truth labels for the record pairs

├── 📃 main.py                     <-- Main script for the project
├── 📃 requirements.txt            <-- Libraries and dependencies
└── 📃 README.md                   <-- Project documentation

```








































