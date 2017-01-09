# TODO: find the way to keep this file named 'io'.
# TODO json <-> utf-8 or w/o:
# http://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file
import json
import importlib

import env


def save_model(model, filepath=None, **json_params):
    filepath = filepath or 'model.json'
    params = model.get_params(deep=False)
    # params <- model._save()
    with open(filepath, 'w') as f:
        json.dump(params, f, **json_params)


def load_model(filepath=None):
    filepath = filepath or 'model.json'
    with open(filepath) as f:
        params = json.load(f)
    if not 'model' in params:
        raise ValueError("missed required field: 'model'")
    model_path = params['model']

    module_path, model_name = model_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, model_name, None)

    if model_class:
        model = model_class()
        model.set_params(**params)
        # model._load(params)
        return model

    raise ValueError("cannot find model '{0}'".format(model_path))