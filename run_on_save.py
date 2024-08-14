import subprocess

import click


@click.command()
@click.option("--file_path", "-f")
def main(file_path):
    subprocess.run(
        [
            "autoflake",
            "--in-place",
            "--remove-unused-variables",
            "--remove-all-unused-imports",
            file_path,
        ],
    )
    subprocess.run(["black", file_path])
    subprocess.run(["isort", file_path])


if __name__ == "__main__":
    main()
