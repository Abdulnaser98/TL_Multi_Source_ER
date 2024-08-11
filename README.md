├── 📁 data                   
│   ├── 📁 raw_data             <-- Unprocessed camera datasets
│   ├── 📁 cleaned_data         <-- Preprocessed camera datasets
│   ├── 📁 help_data            <-- Auxiliary data used during preprocessing
│   ├── 📁 linkage_tasks        <-- Linkage tasks for each pair of sources, with the first two columns containing record IDs
│   ├── 📁 linkage_tasks_labeled<-- Labeled linkage tasks generated using active learning
│   └── 📁 ground_truth_data    <-- Ground truth data for the linkage tasks

├── 📁 results                  
│   ├── 📁 figures              
│   │   ├── 📁 LDA                                  <-- LDA Visualizations (Wordcloud, Stacked Bar Chart)
│   │   ├── Coherence_scores_vs_num_topics.jpg      <-- Coherence scores for different topics
│   │   └── Silhouette_scores_vs_num_topics.jpg     <-- Silhouette scores for different topics
│   ├── 📁 lda_output                                <-- LDA model experiments
│   │   └── lda_model_0.pkl
│   │   └── lda_model_1.pkl
│   │   └── lda_model_2.pkl
│   │   └── ...
│   └── 📃 lda_explanation.txt  <-- Explanation of LDA preprocessing and results

├── 📁 notebooks              <-- Jupyter notebooks
│   ├── Bertopic.ipynb        <-- Implementation of the pretrained "Bertopic" model
│   ├── lda.ipynb             <-- Implementation of LDA
│   ├── lda_evaluation.ipynb  <-- Evaluation of the LDA model
│   └── tfidf.ipynb           <-- Implementation of TF-IDF + clustering

├── 📁 meta_tl                <-- Project code base
│   ├── statistical_tests     <-- Implementations of statistical methods for comparing linkage task distributions
│   ├── graph_clustering.py   <-- Methods for graph clustering of linkage tasks
│   ├── model_selection.py    <-- Methods for selecting linkage tasks from clusters
│   ├── preprocessing.py      <-- Preprocessing scripts
│   └── active_learning.py    <-- Active learning methods for labeling selected tasks

├── 📃 main.py                <-- Algorithm comparison script
├── 📃 requirements.txt       <-- Configuration file for dependencies
└── 📃 README.md              <-- Project documentation
