#!/usr/bin/env python3

import bz2
import gzip
import json
import re
import sys

import click
import h5py

from mtnlion.comsol import *

template_config = [{
    'model': 'model_name',
    'variant': 'variant_name',
    'csv_path': 'comsol_solution',
    'parameters_file': 'params.xlsx',
    'parameters_file_format': 'UCCS_xlsx',
    'physical_domain': ['j.csv', 'phie.csv', 'phis.csv', 'ce.csv'],
    'pseudo_domain': ['cs.csv'],
    'physical_bounds': {'x': [0, 1, 2, 3]},
    'pseudo_bounds': {'x': [0, 1, 1.5, 2.5], 'y': [0, 1]},
    'physical_subdomains': {'neg': {'x': [0, 1]}, 'sep': {'x': [1, 2]}, 'pos': {'x': [2, 3]}, },
    'pseudo_subdomains': {'neg': {'x': [0, 1]}, 'sep': {'x': [1, 1.5]}, 'pos': {'x': [1.5, 2.5]}, },
}]


def fix_boundaries2(data: np.ndarray, boundaries: Union[float, List[int], np.ndarray], num_dim) \
        -> Union[None, np.ndarray]:
    """
    When COMSOL outputs data from the reference model there are two solutions at every internal boundary, which causes
    COMSOL to have repeated domain values; one for the right and one for the left of the boundary. If there is only one
    internal boundary on the variable mesh at a given time, then a duplicate is added.

    :param mesh: x data to use to correct the y data
    :param data: in 2D, this would be the y data
    :param boundaries: internal boundaries
    :return: normalized boundaries to be consistent
    """
    mesh = data[:, 0:num_dim].flatten()
    logger.debug('Fixing boundaries: {}'.format(boundaries))
    b_indices = np.searchsorted(mesh, boundaries)

    if not len(b_indices):
        return data

    for x in b_indices[::-1]:
        if mesh[x] != mesh[x + 1]:  # add boundary
            logger.debug('Adding boundary, copying {x} at index {i}'.format(x=mesh[x], i=x))
            data = np.insert(data, x, data[x], axis=0)

    return data


def opener(filename):
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    elif filename.endswith('.bz2'):
        return bz2.open(filename, 'rt')
    else:
        return open(filename, 'rt')


def append_path(path, file_list):
    return [os.path.join(path, i) for i in file_list]


def collect_time_vector(file, N):
    with opener(file) as f:
        data = [next(f) for _ in range(N)] # read first N lines
        for row in data:
            if 't=' in row:
                return np.array([float(s) for s in re.split('[=,]', row) if s.replace('.','',1).isdigit()])


def find_time_vector(filelist, N):
    data = np.array([collect_time_vector(file, N) for file in filelist])

    for i, d in enumerate(data[:-1, :]):
        if not np.allclose(data[i, :], data[i+1, :]):
            logger.error('Time vectors do not match in {} and {}'.format(filelist[i], filelist[i+1]))
            raise ValueError

    return data[0]


def find_mesh(data, num_dims):
    mesh = None
    old_k = None
    for k, v in data.items():
        if mesh is None:
            old_k = k
            mesh = v[:, 0:num_dims]
            continue

        if not np.allclose(mesh, v[:, 0:num_dims]):
            logger.error('Mesh does not match in {} and {}'.format(old_k, k))
            raise ValueError

    return mesh


# TODO: settable tolerances
# TODO: Documentation
def organize(file_coords, dofs):
    transform = []
    for i in dofs:
        ind1 = np.where(np.abs(file_coords[:, 0] - i[0]) <= 1e-5)
        ind2 = np.where(np.abs(file_coords[:, 1] - i[1]) <= 1e-5)
        if len(ind1[0]) > 0 and len(ind2[0]) > 0:
            transform.append(np.intersect1d(ind1, ind2)[0])
            if len(np.intersect1d(ind1, ind2)) > 1:
                raise ValueError('Too many matching indices')
        else:
            raise ValueError('Missing indices, check tolerances')
    return transform


@click.command()
@click.option('--model', '-M', nargs=1, type=str, help='model name')
@click.option('--template', '-T', flag_value=True, help='create template model config file specified by output')
@click.option('--config', '-c', nargs=1, type=str, help='use the specified config file')

@click.option('--append', 'append', flag_value=True, help='Append model to file')
@click.option('--critical', 'loglevel', flag_value=logging.CRITICAL, help='Set log-level to CRITICAL')
@click.option('--error', 'loglevel', flag_value=logging.ERROR, help='Set log-level to ERROR')
@click.option('--warn', 'loglevel', flag_value=logging.WARNING, help='Set log-level to WARNING')
@click.option('--info', 'loglevel', flag_value=logging.INFO, help='Set log-level to INFO', default=True)
@click.option('--debug', 'loglevel', flag_value=logging.DEBUG, help='Set log-level to DEBUG')
@click.argument('output', type=click.Path(writable=True, resolve_path=True))
def main(output: Union[click.utils.LazyFile, str],
         loglevel: Union[None, int], append, model, template, config) -> Union[None, int]:
    """
    Convert COMSOL CSV files to npz.

    Create a numpy zip (npz) with variables corresponding to the csv file names.
    Each variable contains the data from the file as a list. Additionally, each
    variable is a key in the main dictionary.
    """

    logging.basicConfig(level=loglevel)

    if template:
        with open(output, 'w') as f:
            json.dump(template_config, f, indent=4, separators=(',', ': '))
        sys.exit(0)

    if not config:
        logger.error('Config file required. See help info.')
        sys.exit(1)

    with open(config) as f:
        cfg = json.load(f)

    cfg = cfg[0]
    main_files = append_path(cfg['csv_path'], cfg['physical_domain'])
    pseudo_files = append_path(cfg['csv_path'], cfg['pseudo_domain'])

    physical_domain_data = loader.collect_files(main_files, format_key=format_name, loader=loader.load_csv_file)
    pseudo_domain_data = loader.collect_files(pseudo_files, format_key=format_name, loader=loader.load_csv_file)

    num_dims = 1
    fixed_main = dict()
    for k, v in physical_domain_data.items():
        fixed_main[k] = fix_boundaries2(v, np.array(cfg['physical_bounds']['x'][1:-1]), num_dims)

    time = find_time_vector(main_files + pseudo_files, 10)
    mesh = find_mesh(fixed_main, num_dims).flatten()
    pseudo_mesh = find_mesh(pseudo_domain_data, 2)

    with h5py.File('testoutput.h5', 'w') as f:
        print(output)
        mdl = f.create_group(cfg['model'])
        mdl['time'] = time

        physical_domain = mdl.create_group('physical_domain')
        pseudo_domain = mdl.create_group('pseudo_domain')

        physical_domain['mesh'] = mesh
        pseudo_domain['mesh'] = pseudo_mesh

        for k, v in fixed_main.items():
            d = physical_domain.create_dataset(k, v[1:].T.shape, dtype=np.float)
            d[...] = v[1:].T
            d.dims[0].label = 't'
            d.dims[1].label = 'x'
            d.dims.create_scale(physical_domain['mesh'], 'normalized x')
            d.dims.create_scale(mdl['time'], 'time')
            d.dims[0].attach_scale(mdl['time'])
            d.dims[1].attach_scale(physical_domain['mesh'])

        for k, v in pseudo_domain_data.items():
            d = pseudo_domain.create_dataset(k, v[2:].T.shape, dtype=np.float)
            d[...] = v[2:].T

        for name in physical_domain_data:
            print(name)


    exit()

    if not input_files:
        logger.error('No CSVs were specified. Aborting.')
        sys.exit(1)

    if append:
        access = 'a'
    else:
        access = 'w'

    if not bound:
        bound = [1, 2]
    else:
        bound = loader.load_csv_file(bound)

    logger.info('Output file: {}'.format(output))
    logger.info('Input file(s): {}'.format(input_files))
    logger.info('dt: {}'.format(dt))
    logger.info('boundaries: {}'.format(bound))

    with h5py.File(output, access) as f:
        mdl = f.create_group(model)
        physical_domain = mdl.create_group('physical_domain')
        pseudo_domain = mdl.create_group('pseudo_domain')

    physical_domain_data = loader.collect_files(main, format_key=format_name, loader=loader.load_csv_file)
    pseudo_domain_data = loader.collect_files(pseudo, format_key=format_name, loader=loader.load_csv_file)


    if 'time_mesh' not in file_data:
        try:
            file_data['time_mesh'] = np.arange(dt[0], dt[1] + dt[2], dt[2])
        except IndexError as ex:
            logger.critical('Either a dt option provided with start and stop time (inclusive) or a csv defining the '
                            'time mesh is required', exc_info=True)
            raise ex

    data = format_2d_data(file_data, boundaries=bound)
    data1 = format_pseudo_dim(file_data, boundaries=bound)
    data['pseudo_mesh'] = data1['pseudo_mesh']
    data['cs'] = data1['cs']
    loader.save_npz_file(output, data)

    logger.info('Conversion completed successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
