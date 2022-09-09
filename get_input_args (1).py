import argparse


def get_input_args():
    parser = argparse.ArgumentParser(description='Process flower images.')
    
    parser.add_argument('--dir', default='ImageClassifier/save_directory', help = 'path to the folder of saved checkpoints', action='store')
    parser.add_argument('--check_file', default='./checkpoint.pth', help = 'path to the folder of saved checkpoints', action='store')
    parser.add_argument('--image_path', default='ImageClassifier/flowers/train/19/image_06149.jpg', type= open, help = 'path to predict image', action='store')
    parser.add_argument('--train_set', default='ImageClassifier/flowers/train', help = 'path to predict image')
    parser.add_argument('--arch', type = str, choices=['vgg16', 'vgg19'], default = 'vgg16')
    parser.add_argument('--learning_rate', default = 0.003)
    parser.add_argument('--epochs', type = int, default = 8)
    parser.add_argument('--hidden_units', type = int, default = 1024, help = 'The number of hidden layers to use')
    parser.add_argument('--device', type = str, choices=['cpu', 'cuda'], default = 'cpu')  
       
                                        
                        
    return parser.parse_args()