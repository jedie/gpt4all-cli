"""
    Allow gpt4all_cli to be executable
    through `python -m gpt4all_cli`.
"""


from gpt4all_cli.cli import cli_app


def main():
    cli_app.main()


if __name__ == '__main__':
    main()
