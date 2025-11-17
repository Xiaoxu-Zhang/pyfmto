import argparse
import sys
from os import listdir
from pathlib import Path

from .io import parse_yaml, dumps_yaml
from ..experiments import Launcher, Reports, list_report_formats, DEFAULT_CONF as EXP_CONF
from ..experiments.utils import list_algorithms, get_alg_kwargs
from ..problems import list_problems, DEFAULT_CONF as PROB_CONF


def update_path():
    current_dir = Path().cwd()
    if 'algorithms' in listdir(current_dir):
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))
    else:
        raise FileNotFoundError(f"'algorithms' folder not found in the current directory '{current_dir}'.")


def main():
    update_path()
    parser = argparse.ArgumentParser(
        description='PyFMTO: Python Library for Federated Many-task Optimization Research'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run experiments')
    run_parser.add_argument(
        '-c', '--config', type=str, default='config.yaml',
        help='Path to configuration file (default: config.yaml)')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument(
        '-c', '--config', type=str, default='config.yaml',
        help='Path to configuration file (default: config.yaml)')

    # List command
    list_parser = subparsers.add_parser('list', help='List available options')
    list_parser.add_argument(
        'name', type=str, choices=['problems', 'algorithms', 'reports'],
        help='Name of the option to list'
    )

    # Show command
    show_parser = subparsers.add_parser('show', help='Show default configurations')
    show_parser.add_argument(
        'name', type=str,
        help="Name of the configuration to show, any things that can be list by the 'list' command"
    )

    args = parser.parse_args()

    if args.command == 'run':
        launcher = Launcher(conf_file=args.config)
        launcher.run()
    elif args.command == 'report':
        reports = Reports(conf_file=args.config)
        reports.generate()
    elif args.command == 'list':
        if args.name == 'problems':
            print('\n'.join(list(parse_yaml(PROB_CONF).keys())))
        elif args.name == 'algorithms':
            list_algorithms(print_it=True)
        elif args.name == 'reports':
            list_report_formats(print_it=True)
    elif args.command == 'show':
        if args.name.lower() in list(map(lambda x: x.lower(), list_problems())):
            conf = parse_yaml(PROB_CONF)
            data = conf.get(args.name.lower())
            if not data:
                print(f"Configuration for {args.name} not found.")
            else:
                print(dumps_yaml(data))
        elif args.name.upper() in list_algorithms():
            kwargs = get_alg_kwargs(args.name)
            print(dumps_yaml(kwargs))
        elif args.name.lower() in list_report_formats():
            conf = parse_yaml(EXP_CONF)
            print(dumps_yaml(conf[args.name]))
        else:
            print(f"Config for {args.name} not found. Please check the available options using 'list' command.")
