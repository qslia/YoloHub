import torch
from model import yolov3


def main():
    x = torch.rand(1, 3, 258, 258)
    model = yolov3()
    y = model.forward(x)
    print(y.shape)


if __name__ == '__main__':
    main()
