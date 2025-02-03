import click
from src.inference_approach1 import main as approach1_main
from src.inference_approach2 import main as approach2_main


@click.command()
@click.option('--approach', type=click.Choice(['1', '2']), default='2', help='Choose the approach to use (1 or 2)')
@click.option('--type', type=click.Choice(['video', 'image']), required=True, help='Specify the type of input (video or image)')
@click.option('--path', type=click.Path(exists=True), required=True, help='Path to the video or image file')
def cli(approach, type, path):
    """License Plate Recognition CLI (plate-ocr)"""
    if approach == '1':
        approach1_main(type, path)
    elif approach == '2':
        approach2_main(type, path)

if __name__ == '__main__':
    cli()