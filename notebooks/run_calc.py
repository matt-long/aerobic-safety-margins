import json
import os
import pathlib
import subprocess
import warnings
from glob import glob

import click
import jupyter_client
import nbformat
import yaml


def get_conda_kernel_cwd(name: str):
    """get the directory of a conda kernel by name"""
    command = ['conda', 'env', 'list', '--json']
    output = subprocess.check_output(command).decode('ascii')
    envs = json.loads(output)['envs']
    for env in envs:
        env = pathlib.Path(env)
        if name == env.stem:
            return env
    else:
        return None


def nb_set_kernelname(file_in, kernel_name, file_out=None):
    """set the kernel name to python3"""
    if file_out is None:
        file_out = file_in
    data = nbformat.read(file_in, as_version=nbformat.NO_CONVERT)
    data['metadata']['kernelspec']['name'] = kernel_name
    nbformat.write(data, file_out)


def nb_get_kernelname(file_in):
    """get the kernel name of a notebook"""
    data = nbformat.read(file_in, as_version=nbformat.NO_CONVERT)
    return data['metadata']['kernelspec']['name']


def nb_clear_outputs(file_in, file_out=None):
    """clear output cells"""
    if file_out is None:
        file_out = file_in
    data = nbformat.read(file_in, as_version=nbformat.NO_CONVERT)

    assert isinstance(data['cells'], list), 'cells is not a list'

    cells = []
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
        cells.append(cell)
    data['cells'] = cells
    nbformat.write(data, file_out)


def nb_execute(notebook_filename, output_dir='.', kernel_name='python3'):
    """
    Execute a notebook.
    see http://nbconvert.readthedocs.io/en/latest/execute_api.html
    """
    import io

    import nbformat
    from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

    # -- open notebook
    with io.open(notebook_filename, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    # config for execution
    ep = ExecutePreprocessor(timeout=None)  # , kernel_name=kernel_name)

    # run with error handling
    try:
        out = ep.preprocess(nb, {'metadata': {'path': './'}})

    except CellExecutionError:
        out = None
        msg = f'Error executing the notebook "{notebook_filename}".\n'
        msg += f'See notebook "{notebook_filename}" for the traceback.\n'
        print(msg)

    finally:
        nb_out = os.path.join(output_dir, os.path.basename(notebook_filename))
        with io.open(nb_out, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f'wrote: {nb_out}')

    return out


def install_conda_kernel(kernel_name):
    """install a conda kernel in a location findable by `nbconvert` etc."""
    path = get_conda_kernel_cwd(kernel_name) / pathlib.Path('share/jupyter/kernels')
    _kernels = jupyter_client.kernelspec._list_kernels_in(path)
    if len(_kernels) > 1:
        raise ValueError(f'expecting to find 1 kernel; found: {_kernels}')
    k, kernel_path = _kernels.popitem()
    jupyter_client.kernelspec.install_kernel_spec(
        kernel_path, kernel_name=kernel_name, user=True, replace=True
    )
    jupyter_client.kernelspec.find_kernel_specs()


def run(notebooks, kernel, stop_on_fail=True):
    """
    Run a list of notebooks.

    Warning: this function presumes that it is being called from
             within the same environment as the kernel of the
             intended notebooks.

    Warning: don't call this function on a notebook from *within* that
             same notebook. That could yield an infinite recursive loop.
    """

    # check kernels
    kernels = {}
    for nb in notebooks:
        assert os.path.exists(nb), f'Notebook not found: {nb}'
        kernels[nb] = nb_get_kernelname(nb)

    if len(set(kernels.values())) > 1:
        warnings.warn(
            f'not all notebooks have the same kernel: {kernels}\nrunning all notebooks with {kernel}'
        )

    cwd = os.getcwd()
    ran_ok = []
    for nb in notebooks:
        print('-' * 80)
        print(f'executing: {nb}')

        # set the kernel name to fool nbconvert into running this
        nb_set_kernelname(nb, kernel_name=kernel)

        # clear output
        nb_clear_outputs(nb)

        # run the notebook
        ok = nb_execute(nb, output_dir=cwd, kernel_name=kernel)
        ran_ok.append(ok)

        # set the kernel back
        nb_set_kernelname(nb, kernel_name=kernels[nb])

        if not ok and stop_on_fail:
            raise RuntimeError(f'failed: {nb}')

    return all(ran_ok)


@click.command()
@click.argument('notebooks', nargs=-1)
@click.option('--kernel', required=True)
def main(notebooks, kernel):
    kernel_specs = jupyter_client.kernelspec.find_kernel_specs()
    print(kernel)
    assert (
        kernel in kernel_specs
    ), f"Kernel '{kernel}' not found in kernelspec.\n Available kernels:\n{list(kernel_specs.keys())}"

    print('running notebooks:')
    for nb in notebooks:
        print(f'- {nb}')
    print()

    run(notebooks, kernel=kernel, stop_on_fail=True)


if __name__ == '__main__':
    main()
