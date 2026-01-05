import argparse
from pathlib import Path
from typing import cast, Literal

from .tools import matched_str_head
from ..experiment import Launcher, Reports, list_report_formats, show_default_conf
from .loaders import add_sources, ConfigLoader, ProblemData, AlgorithmData


def main():
    add_sources([str(Path().cwd())])

    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        '-c', '--config', type=str, default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser = argparse.ArgumentParser(
        description='PyFMTO: Python Library for Federated Many-task Optimization Research'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    subparsers.add_parser('run', parents=[global_parser], help='Run experiments')

    # Report command
    subparsers.add_parser('report', parents=[global_parser], help='Generate reports')

    # List command
    list_parser = subparsers.add_parser('list', parents=[global_parser], help='List available options')
    list_parser.add_argument(
        'name', type=str, help='Name of the option to list'
    )

    # Show command
    show_parser = subparsers.add_parser('show', parents=[global_parser], help='Show default configurations')
    show_parser.add_argument(
        'name', type=str,
        help="Name of the configuration to show, any things that can be list by the 'list' command"
    )
    args = parser.parse_args()

    conf = ConfigLoader(config=args.config)

    if args.command == 'run':
        launcher = Launcher(conf=conf.launcher)
        launcher.run()
    elif args.command == 'report':
        reports = Reports(conf=conf.reporter)
        reports.generate()
    elif args.command == 'list':
        full_name = matched_str_head(args.name, ['problems', 'algorithms', 'reports'])
        if full_name == 'problems':
            conf.show_sources(cast(Literal['algorithms', 'problems'], full_name), print_it=True)
        elif full_name == 'algorithms':
            conf.show_sources(cast(Literal['algorithms', 'problems'], full_name), print_it=True)
        elif full_name == 'reports':
            list_report_formats(print_it=True)
    elif args.command == 'show':
        t, v = args.name.split('.')
        full_name = matched_str_head(t, ['problems', 'algorithms', 'reports'])
        if full_name == 'problems':
            prob = conf.problems.get(v, ProblemData(v, []))
            print(prob.params_yaml)
        elif full_name == 'algorithms':
            alg = conf.algorithms.get(v, AlgorithmData(v, []))
            print(alg.params_yaml)
        elif full_name == 'reports':
            show_default_conf(v)
        else:
            print(f"No matched group for {t}.")
