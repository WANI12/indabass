import json
from pathlib import Path

nb_path = Path('Starter_Notebook.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))
ns = {}
for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if source.strip():
            exec(compile(source, f'<cell {i}>', 'exec'), ns, ns)

print('Notebook execution completed.')
print('submission.csv exists:', Path('submission.csv').exists())
if Path('submission.csv').exists():
    import pandas as pd
    print(pd.read_csv('submission.csv').head().to_string(index=False))
