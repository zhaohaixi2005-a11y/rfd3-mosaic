"""Progressive CLI help without changing accepted commands or arguments."""

from __future__ import annotations

import argparse


class _WorkflowHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, show_all: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_all = show_all

    def _visible(self, action: argparse.Action) -> bool:
        return self.show_all or not getattr(action, "mosaic_advanced", False)

    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(
            usage,
            [action for action in actions if self._visible(action)],
            groups,
            prefix,
        )

    def add_arguments(self, actions):
        super().add_arguments([action for action in actions if self._visible(action)])

    def _iter_indented_subactions(self, action):
        for subaction in super()._iter_indented_subactions(action):
            if self._visible(subaction):
                yield subaction


class _FullHelpAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.show_all_help = True
        parser.print_help()
        parser.exit()


class WorkflowArgumentParser(argparse.ArgumentParser):
    """Keep optional expert controls discoverable through ``--help-all``."""

    show_all_help = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--help-all",
            action=_FullHelpAction,
            dest=argparse.SUPPRESS,
            help="Show all commands or options, including advanced controls.",
        )

    def _get_formatter(self):
        return _WorkflowHelpFormatter(prog=self.prog, show_all=self.show_all_help)


def primary_options(parser: argparse.ArgumentParser, destinations: set[str]) -> None:
    """Affects display only; hidden arguments retain their existing parsing."""
    for action in parser._actions:
        if action.option_strings and action.dest not in destinations | {
            "help",
            argparse.SUPPRESS,
        }:
            action.mosaic_advanced = True


def primary_commands(action: argparse._SubParsersAction, names: set[str]) -> None:
    for choice in action._choices_actions:
        choice.mosaic_advanced = choice.dest not in names
