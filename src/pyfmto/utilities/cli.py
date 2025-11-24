import argparse
import sys
from pathlib import Path

from .tools import matched_str_head
from ..experiments import Launcher, Reports, list_report_formats, show_default_conf
from .loaders import list_problems, load_algorithm, list_algorithms, load_problem


def update_path():
    current_dir = Path().cwd()
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))


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
        'name', type=str, help='Name of the option to list'
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
        full_name = matched_str_head(args.name, ['problems', 'algorithms', 'reports'])
        if full_name == 'problems':
            list_problems(print_it=True)
        elif full_name == 'algorithms':
            list_algorithms(print_it=True)
        elif full_name == 'reports':
            list_report_formats(print_it=True)
    elif args.command == 'show':
        t, v = args.name.split('.')
        full_name = matched_str_head(t, ['problems', 'algorithms', 'reports'])
        if full_name == 'problems':
            prob = load_problem(v)
            print(prob.params_yaml)
        elif full_name == 'algorithms':
            alg = load_algorithm(v)
            print(alg.params_yaml)
        elif full_name == 'reports':
            show_default_conf(v)
        else:
            print(f"No matched group for {t}.")
