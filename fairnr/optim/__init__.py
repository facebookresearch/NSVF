import importlib
import os

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        optimizer_name = file[: file.find(".py")]
        importlib.import_module(
            "fairnr.optim." + optimizer_name
        )