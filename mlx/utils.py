import os
import shutil


def make_dir(path, check_empty=False, force_empty=False, use_dirname=False):
    """Make a local directory.

    Args:
        path: path to directory
        check_empty: if True, check that directory is empty
        force_empty: if True, delete files if necessary to make directory
            empty
        use_dirname: if path is a file, use the the parent directory as path

    Raises:
        ValueError if check_empty is True and directory is not empty
    """
    directory = path
    if use_dirname:
        directory = os.path.abspath(os.path.dirname(path))

    if force_empty and os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)

    if check_empty and any(os.scandir(directory)):
        raise ValueError(
            '{} needs to be an empty directory!'.format(directory))
