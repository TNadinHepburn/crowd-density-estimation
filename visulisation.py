import matplotlib.pyplot as plt
from datetime import datetime
def losslines(results):
    plt.plot(results)
    plt.show()


def inputsOutputsGT(img,gt,pred):

    plt.subplot(1,3,1)
    plt.plot(img)
    plt.title('Input Image')

    plt.subplot(1,3,2)
    plt.plot(gt)
    plt.title('Ground Truth')

    plt.subplot(1,3,3)
    plt.plot(pred)
    plt.title('Predicted Output')

    plt.savefig(f'Results/{datetime.now().strftime("%m%d")}/{datetime.now().strftime("%H%M%S")}.png')
    plt.show()
