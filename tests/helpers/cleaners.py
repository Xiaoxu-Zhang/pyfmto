from pyfmto.utilities import colored

__all__ = ['clear_problems_cache', 'remove_temp_files']


def clear_problems_cache():
    import sys
    to_delete = [name for name in sys.modules if name.startswith('problems')]
    for name in to_delete:
        del sys.modules[name]
        print(f"Problems cache [{colored(name, 'red')}] cleared.")


def remove_temp_files():
    from pathlib import Path
    import shutil
    if Path('problems').exists():
        clear_problems_cache()
    for file in ['algorithms', 'problems', 'out']:
        shutil.rmtree(file, ignore_errors=True)
    Path('config.yaml').unlink(missing_ok=True)
    Path('setting.yaml').unlink(missing_ok=True)
