# Project Setup Instructions

## Python Version
Python 3.11.8

## Installation Steps
1. Create virtual environment:
```
   python -m venv xray-project
```

2. Activate virtual environment:
```
   .\xray-project\Scripts\Activate.ps1
```

3. Install dependencies:
```
   pip install -r requirements.txt
```

4. Run the app:
```
   streamlit run app.py
```

## Requirements
- Visual C++ Redistributable 2015-2022
```

## 3. Add a .gitignore File

Create `.gitignore` to avoid committing unnecessary files:
```
xray-project/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/