import argparse
from pathlib import Path
from typing import cast

from pyfmto import load_problem
from pyfmto.utilities.loaders import list_components, load_algorithm

from ..core.typing import TComponentNames
from ..experiment import ConfigLoader, Launcher, Reporter, list_report_formats, show_default_conf
from .tools import add_sources, matched_str_head


def main() -> None:
    add_sources([str(Path.cwd())])

    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        '-c', '--config', type=str, default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser = argparse.ArgumentParser(
        description='PyFMTO: Python Library for Federated Many-task Optimization Research'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run/Report command
    subparsers.add_parser('run', parents=[global_parser], help='Run experiments')
    subparsers.add_parser('report', parents=[global_parser], help='Generate reports')

    # List command
    list_p = subparsers.add_parser('list', parents=[global_parser], help='List available options')
    list_p.add_argument(
        'name', type=str, help='Name of the option to list'
    )

    # Show command
    show_p = subparsers.add_parser('show', parents=[global_parser], help='Show default configurations')
    show_p.add_argument(
        'name', type=str,
        help="Name of the configuration to show, any things that can be list by the 'list' command"
    )
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    conf = ConfigLoader(config=args.config)

    handlers = {
        "run": lambda: _handle_run(conf),
        "report": lambda: _handle_report(conf),
        "list": lambda: _handle_list(args, conf),
        "show": lambda: _handle_show(args, conf),
    }
    handlers[args.command]()


def _handle_run(conf: ConfigLoader):
    launcher = Launcher(conf=conf.launcher)
    launcher.run()


def _handle_report(conf: ConfigLoader):
    reporter = Reporter(conf=conf.reporter)
    reporter.report()


def _handle_list(args, conf: ConfigLoader):
    full_name = matched_str_head(args.name, ['problems', 'algorithms', 'reports'])
    if full_name in ('problems', 'algorithms'):
        list_components(cast(TComponentNames, full_name), conf.sources, print_it=True)
    elif full_name == 'reports':
        list_report_formats(print_it=True)
    else:
        print(f"Unknown list option: {args.name}")


def _handle_show(args, conf: ConfigLoader):
    if '.' not in args.name:
        print("Error: name must be in format 'group.item' (e.g., problems.MyProblem)")
        return

    t, v = args.name.split('.', 1)
    full_name = matched_str_head(t, ['problems', 'algorithms', 'reports'])

    if full_name == 'problems':
        print(load_problem(v, sources=conf.sources).params_yaml)
    elif full_name == 'algorithms':
        print(load_algorithm(v, sources=conf.sources).params_yaml)
    elif full_name == 'reports':
        show_default_conf(v)
    else:
        print(f"No matched group for '{t}'. Available: problems, algorithms, reports")
