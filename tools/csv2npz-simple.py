#!/usr/bin/env python3
import sys

import click

from mtnlion.comsol import *


@click.command()
@click.option("--critical", "loglevel", flag_value=logging.CRITICAL, help="Set log-level to CRITICAL")
@click.option("--error", "loglevel", flag_value=logging.ERROR, help="Set log-level to ERROR")
@click.option("--warn", "loglevel", flag_value=logging.WARNING, help="Set log-level to WARNING")
@click.option("--info", "loglevel", flag_value=logging.INFO, help="Set log-level to INFO", default=True)
@click.option("--debug", "loglevel", flag_value=logging.DEBUG, help="Set log-level to DEBUG")
@click.argument("output", type=click.Path(writable=True, resolve_path=True))
@click.argument("input_files", nargs=-1, type=click.Path(exists=True, readable=True, resolve_path=True))
def main(
    input_files: List[str], output: Union[click.utils.LazyFile, str], loglevel: Union[None, int]
) -> Union[None, int]:
    """
    Convert COMSOL CSV files to npz.

    Create a numpy zip (npz) with variables corresponding to the csv file names.
    Each variable contains the data from the file as a list. Additionally, each
    variable is a key in the main dictionary.
    """

    logging.basicConfig(level=loglevel)

    if not input_files:
        logger.error("No CSVs were specified. Aborting.")
        sys.exit(1)

    logger.info("Output file: {}".format(output))
    logger.info("Input file(s): {}".format(input_files))

    file_data = loader.collect_files(input_files, format_key=format_name, loader=loader.load_csv_file)
    loader.save_npz_file(output, file_data)

    logger.info("Conversion completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
