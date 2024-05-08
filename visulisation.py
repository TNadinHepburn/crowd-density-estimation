import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
def losslines(results):

    plt.plot(results)

    plt.xlabel("Epochs")
    plt.ylabel("Loss(MAE)")
    plt.ylim(min(results) - 0.05, max(results) + 0.05)
    plt.yticks(np.arange(min(results) - 0.05, max(results) + 0.05, 0.01))
    plt.savefig(f'Results/graphs/TRANSFORMER_MAE.png')
    plt.show()


def inputsOutputsGT(img,gt,pred):

    print('bef img',img.shape)
    print('bef gt',gt.shape)
    img = img.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    gt = gt.cpu().detach().numpy().squeeze()

    # print('aft img',img.shape)
    # print('aft gt',gt.shape)

    plt.figure(figsize=(15, 5))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('Input Image')

    plt.subplot(1,3,2)
    plt.imshow(gt, cmap='jet')
    plt.title('Ground Truth')

    plt.subplot(1,3,3)
    plt.imshow(abs(pred), cmap='jet')
    plt.title('Predicted Output')

    plt.tight_layout()
    plt.savefig(f'Results/{datetime.now().strftime("%m%d")} {datetime.now().strftime("%H%M%S")}.png')
    

if __name__ == '__main__':
    results_trans =     [0.483, 0.47, 0.48, 0.476, 0.476, 0.476, 0.479, 0.48, 0.471, 0.478, 0.484, 0.475, 0.484, 0.489, 0.479, 0.476, 0.479, 0.482, 0.484, 0.482, 0.469, 0.481, 0.482, 0.481]
    results_cnn = [0.305, 0.312, 0.31, 0.303, 0.307, 0.306, 0.311, 0.311, 0.313, 0.303, 0.305, 0.309, 0.301, 0.308, 0.307, 0.304, 0.305, 0.312, 0.306, 0.306, 0.311, 0.311, 0.304, 0.309]    
    losslines(results_trans)