# mlops

Coursework for MLOps course of DTU

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

### Main usage:

- **Train**: python <train_script> (e.g. `python src/mlops/train.py`)
- **Evaluate**: python <evaluate_script> <model_checkpoint> (e.g. `python src/mlops/evaluate.py models/model.pth`)
- **Visualize**: python <visualize_script> <model_checkpoint> (e.g. `python src/mlops/visualize.py models/model.pth`) 
- **View model structure**: python <model_script> (e.g. `python src/mlops/model.py`)

### OR use *invoke* (check tasks.py file):

- **Train**: `invoke train`
- **Preprocess data**: `invoke preprocess-data`
- **Install requirements**: `invoke requirements`
- **Install development requirements**: `invoke dev-requirements`
- **Build Docker images**: `invoke docker-build`
- **Build documentation**: `invoke build-docs`
- **Serve documentation**: `invoke serve-docs`
- **Create environment**: `invoke create-environment`
- **Run tests**: `invoke test`
- **Git commit and push**: `invoke git --message "your commit message"`
- **Setup Conda environment**: `invoke conda --name "your_env_name"`