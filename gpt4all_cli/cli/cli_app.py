"""
    CLI for usage
"""
import multiprocessing
import time

from gpt4all import GPT4All
import logging
import sys
from pathlib import Path

import rich_click as click
from bx_py_utils.path import assert_is_file
from rich import print  # noqa
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table
from rich_click import RichGroup

import gpt4all_cli
from gpt4all_cli import constants, __version__


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
    table = Table(title="Star Wars Movies")
    console = Console()
    skip_keys = {'order', 'url', 'md5sum', 'name'}
    with console.status("Fetch..."):
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
@click.option(
    "--model",
    # default='mistral-7b-openorca.Q4_0.gguf',
    default='orca-mini-3b-gguf2-q4_0.gguf',
    # default='rift-coder-v0-7b-q4_0.gguf',
)
@click.option("--max-tokens", type=click.IntRange(1, 9999), default=100)
@click.option("--n_batch", type=click.IntRange(1, 9999), default=multiprocessing.cpu_count())
@click.argument('prompt', nargs=-1)
def ask(prompt, model, max_tokens, n_batch):
    """
    Ask GPT4all something...

    e.g.:
        ./cli.py ask What is Python?

    https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-bindings/python
    """
    prompt = ' '.join(prompt)
    if not prompt:
        print('Please aks me something, e.g.:')
        print('./cli.py ask What is Python?')
        return

    console = Console()
    console.print('\n')

    console.print(f'Use {model=}...')
    model = GPT4All(model, verbose=True)

    kwargs = dict(
        max_tokens=max_tokens, n_batch=n_batch,
    )

    console.print(Pretty(kwargs))

    console.rule(f'[bold red]{prompt}')
    start_time = time.monotonic()

    generator = model.generate(prompt, streaming=True, **kwargs)

    try:
        for token in generator:
            console.print(token, end='')
    except KeyboardInterrupt:
        console.print('...')

    duration = time.monotonic() - start_time
    console.print()
    console.rule(f'Duration: {duration:.1f}sec')
    console.print()


cli.add_command(ask)


def main():
    print(f'[bold][green]gpt4all_cli[/green] v[cyan]{__version__}')

    # Execute Click CLI:
    cli.name = './cli.py'
    cli()
