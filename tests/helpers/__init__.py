
def remove_temp_files():
    import shutil
    for file in ['algorithms', 'problems', 'out']:
        shutil.rmtree(file, ignore_errors=True)
