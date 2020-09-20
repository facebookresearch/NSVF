import importlib
import os

# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if not file.startswith('_') and not file.startswith('.') and (file.endswith('.py') or os.path.isdir(path)):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('fairnr.models.' + model_name)
