import time

from bx_py_utils.humanize.time import human_timedelta
from gpt4all import GPT4All, LLModel
from rich import print  # noqa
from rich.console import Console
from rich.table import Table


class GptChat:
    """
    https://docs.gpt4all.io/gpt4all_python.html
    """

    def __init__(self, *, initial_prompt: str, model_name: str, max_tokens: int, cpu_count: int, temperature: float):
        self.console = Console()
        self.console.print('\n')

        self.console.print(f'Use {model_name=}...')
        gpt4all = GPT4All(model_name, n_threads=cpu_count, verbose=True)
        model: LLModel = gpt4all.model
        thread_count = model.thread_count()
        self.console.print(f'Using {thread_count} threads...')

        table = Table(title='GPT4All info')
        table.add_column('Parameter')
        table.add_column('Value')
        table.add_row('model', model_name)
        table.add_row('Thread count', str(thread_count))
        table.add_row('Temperature', str(temperature))
        table.add_row('Max tokens', str(max_tokens))
        table.add_row('System prompt', repr(gpt4all.config['systemPrompt']))
        table.add_row('Prompt template', repr(gpt4all.config['promptTemplate']))
        self.console.print(table)

        self.generate_kwargs = dict(max_tokens=max_tokens, temp=temperature)

        self.chat_session = gpt4all.chat_session().__enter__()

        if initial_prompt:
            self.ask(prompt=initial_prompt)

    def loop(self):
        while True:
            prompt = self.console.input('You: ')
            if not prompt:
                self.console.print('\nBye!\n')
                self.chat_session.__exit__(None, None, None)
                return
            self.ask(prompt=prompt)

    def ask(self, *, prompt):
        self.console.rule(f'[bold red]{prompt}')
        start_time = time.monotonic()

        generator = self.chat_session.generate(prompt, streaming=True, **self.generate_kwargs)

        try:
            for token in generator:
                self.console.print(token, end='')
        except KeyboardInterrupt:
            self.console.print('...')

        duration = time.monotonic() - start_time
        self.console.print()
        self.console.rule(f'Duration: {human_timedelta(duration)}')
        self.console.print()
