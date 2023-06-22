import argparse

def py_config_to_args(**kwargs):
    args = argparse.Namespace()
    for k, v in kwargs.items():
        exec(f"args.{k}={v}")
    return args