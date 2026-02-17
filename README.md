# MSProject_MP

Personal research project focused on asset pricing, factor models, and replication of "Shrinking the Cross-Section" (Kozak, Nagel, Santosh, 2020).

## File Structure
MSProject_MP/
├── venv/
│
├── Replication-Shrinking-the-Cross-Section/   # open source code with my edits 
│   ├── Data/                             # contains all data analyzed (25 portfolios, etc)
│   ├── results_export/                   # plots
│   ├── scs_main.py                       # run this in terminal for overall output 
│   ├── SCS_L2est.py                      # estimators & logic 
│   ├── load_ff25.py                      # edits and parses FF25 portfolio data 
│   ├── load_managed_portfolios.py        
│   ├── load_ff_anomalies.py            
│   ├── cross_validate.py              
│   ├── utils.py                          
│   └── ...                         
│
├── scratch/                              # estimators and my code from scratch for FF25 
├── my_notebooks/                         
└── .gitignore                            
