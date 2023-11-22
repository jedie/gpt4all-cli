"""
    CLI for usage
"""
import logging
import multiprocessing
import sys
from pathlib import Path

import rich_click as click
from bx_py_utils.path import assert_is_file
from cli_base.cli_tools.verbosity import OPTION_KWARGS_VERBOSE, setup_logging
from gpt4all import GPT4All
from rich import print  # noqa
from rich.console import Console
from rich.table import Table
from rich_click import RichGroup

import gpt4all_cli
from gpt4all_cli import __version__, constants
from gpt4all_cli.gpt import GptChat


logger = logging.getLogger(__name__)


PACKAGE_ROOT = Path(gpt4all_cli.__file__).parent.parent
assert_is_file(PACKAGE_ROOT / 'pyproject.toml')

OPTION_ARGS_DEFAULT_TRUE = dict(is_flag=True, show_default=True, default=True)
OPTION_ARGS_DEFAULT_FALSE = dict(is_flag=True, show_default=True, default=False)
ARGUMENT_EXISTING_DIR = dict(
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path)
)
ARGUMENT_NOT_EXISTING_DIR = dict(
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        readable=False,
        writable=True,
        path_type=Path,
    )
)
ARGUMENT_EXISTING_FILE = dict(
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path)
)


class ClickGroup(RichGroup):  # FIXME: How to set the "info_name" easier?
    def make_context(self, info_name, *args, **kwargs):
        info_name = './cli.py'
        return super().make_context(info_name, *args, **kwargs)


@click.group(
    cls=ClickGroup,
    epilog=constants.CLI_EPILOG,
)
def cli():
    pass


@click.command()
def version():
    """Print version and exit"""
    # Pseudo command, because the version always printed on every CLI call ;)
    sys.exit(0)


cli.add_command(version)


@click.command()
def list_models():
    table = Table(title='GPT4All Models')
    console = Console()
    skip_keys = {'order', 'url', 'md5sum', 'name'}
    with console.status('Fetch...'):
        models = GPT4All.list_models()
        keys = None
        for model in models:
            if not keys:
                keys = sorted(model.keys() - skip_keys)
                for key in keys:
                    table.add_column(key)
            values = [model.get(key) for key in keys]
            table.add_row(*values)

    print(table)


cli.add_command(list_models)


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument('prompt', nargs=-1)
@click.option(
    "--model",
    # default='wizardlm-13b-v1.2.Q4_0.gguf',  # Big and slow on CPU ;)
    # default='mistral-7b-openorca.Q4_0.gguf',
    default='orca-mini-3b-gguf2-q4_0.gguf',
    # default='rift-coder-v0-7b-q4_0.gguf',
)
@click.option("--max-tokens", type=click.IntRange(1, 9999), default=100)
@click.option("--cpu-count", type=click.IntRange(1, 9999), default=multiprocessing.cpu_count())
@click.option("--temperature", type=click.FloatRange(0, 2), default=0)
@click.option('-v', '--verbosity', **OPTION_KWARGS_VERBOSE)
def chat(prompt, model, max_tokens, cpu_count, temperature, verbosity: int):
    """
    Chat with GPT4all

    https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-bindings/python
    """
    setup_logging(verbosity=verbosity)
    chat = GptChat(
        initial_prompt=' '.join(prompt),
        model_name=model,
        max_tokens=max_tokens,
        cpu_count=cpu_count,
        temperature=temperature,
    )
    chat.loop()


cli.add_command(chat)


def main():
    print(f'[bold][green]gpt4all_cli[/green] v[cyan]{__version__}')

    # Execute Click CLI:
    cli.name = './cli.py'
    cli()
