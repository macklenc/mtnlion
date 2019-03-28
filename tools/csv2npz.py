#!/usr/bin/env python3
import sys

import click

from mtnlion.comsol import *


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
@click.option('--dt', '-t', nargs=3, type=float, help='[start time stop time delta time]')
@click.option('--bound', '-b', type=click.Path(exists=True, readable=True, resolve_path=True),
              help='file containing boundaries, defaults to [1, 2]')
@click.option('--critical', 'loglevel', flag_value=logging.CRITICAL, help='Set log-level to CRITICAL')
@click.option('--error', 'loglevel', flag_value=logging.ERROR, help='Set log-level to ERROR')
@click.option('--warn', 'loglevel', flag_value=logging.WARNING, help='Set log-level to WARNING')
@click.option('--info', 'loglevel', flag_value=logging.INFO, help='Set log-level to INFO', default=True)
@click.option('--debug', 'loglevel', flag_value=logging.DEBUG, help='Set log-level to DEBUG')
@click.argument('output', type=click.Path(writable=True, resolve_path=True))
@click.argument('input_files', nargs=-1, type=click.Path(exists=True, readable=True, resolve_path=True))
def main(input_files: List[str], output: Union[click.utils.LazyFile, str],
         dt: List[float], loglevel: Union[None, int], bound: Union[None, str]) -> Union[None, int]:
    """
    Convert COMSOL CSV files to npz.

    Create a numpy zip (npz) with variables corresponding to the csv file names.
    Each variable contains the data from the file as a list. Additionally, each
    variable is a key in the main dictionary.
    """

    logging.basicConfig(level=loglevel)

    if not input_files:
        logger.error('No CSVs were specified. Aborting.')
        sys.exit(1)

    if not bound:
        bound = [1, 2]
    else:
        bound = loader.load_csv_file(bound)

    logger.info('Output file: {}'.format(output))
    logger.info('Input file(s): {}'.format(input_files))
    logger.info('dt: {}'.format(dt))
    logger.info('boundaries: {}'.format(bound))

    file_data = loader.collect_files(input_files, format_key=format_name, loader=loader.load_csv_file)
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
