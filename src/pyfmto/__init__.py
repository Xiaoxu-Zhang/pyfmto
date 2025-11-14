__version__ = "0.0.1"

import argparse
from pyfmto.framework import (
    export_problem_config,
    export_launcher_config,
    export_reporter_config,
    export_algorithm_config,
)
from pyfmto.experiments.utils import list_algorithms, load_algorithm
from pyfmto.experiments import Launcher, Reports
from pyfmto.problems import list_problems, load_problem


__all__ = [
    'list_problems',
    'list_algorithms',
    'load_problem',
    'load_algorithm',
    'export_problem_config',
    'export_launcher_config',
    'export_reporter_config',
    'export_algorithm_config',
]


def main():
    parser = argparse.ArgumentParser(
        description='PyFMTO: Python Library for Federated Many-task Optimization Research'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run experiments')
    run_parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file (default: config.yaml)')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file (default: config.yaml)')

    args = parser.parse_args()

    if args.command == 'run':
        launcher = Launcher(conf_file=args.config)
        launcher.run()
    elif args.command == 'report':
        reports = Reports(conf_file=args.config)
        reports.generate()


if __name__ == '__main__':
    main()  # pragma: no cover
