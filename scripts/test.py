import argparse

parser = argparse.ArgumentParser(description='Train SocialEgoNet on JPL-P4S')
parser.add_argument('--trainset_path', type=str, required=True,
                    help='Trainset Path')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
args = parser.parse_args()
