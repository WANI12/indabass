#INDABA SOUTH SUDAN CHALLENGE 

Build a data visualisation and prediction MML and MLM model for drought.


Replace `README.md` with the following setup guide. Update `<your-repository-url>` to the repository’s clone URL.

````markdown
# Indaba South Sudan Challenge

Build data visualisation and machine-learning models to predict drought.

## Run from a clone or fork

### 1. Get the project

To clone the repository:

```powershell
git clone <your-repository-url>
cd indabass
```

To work from a fork, fork the repository on GitHub, then clone your fork using its URL.

### 2. Set up Python

Python 3.10 or newer is recommended. From the project folder, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this command for the current terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If the repository contains `requirements.txt`:

```powershell
py -m pip install -r requirements.txt
```

Otherwise, install the packages required by the notebook. For example:

```powershell
py -m pip install jupyter pandas
```

Add any other packages imported by `Starter_Notebook.ipynb`.

### 4. Run the prediction notebook

Make sure `Starter_Notebook.ipynb` and any required data files are in the project folder. Then run:

```powershell
py run_notebook_exec.py
```

The script executes the notebook’s code cells and reports whether `submission.csv` was created. If it exists, the script prints a preview of the file.

To open and run the notebook interactively instead:

```powershell
py -m pip install notebook
jupyter notebook Starter_Notebook.ipynb
```

## Troubleshooting

- If a package import fails, install that package in the activated virtual environment with `py -m pip install <package-name>`.
- If the notebook cannot find a data file, check that the file is present and that its path in the notebook is correct.
- Run commands from the project folder so relative file paths resolve correctly.
````