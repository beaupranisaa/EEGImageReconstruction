import numpy as np
import torch
import os

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu = f'cuda:{np.argmax(memory_available)}'
    return gpu

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

def do_plot(d_losses, g_losses):
    plt.figure(figsize=(25,5))
#     clear_output(wait=True)
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.title('GAN loss')
    plt.legend()
    plt.show()
    
def random_2D_noise(m,n):
    """
    Random an 2d array of random noise
    =======================
    m = # of samples
    n = # of features
    """
    z     = np.random.uniform(-1, 1, size=(m,n))
    z     = torch.from_numpy(z).float()
    return z

def display_img(img, epoch,labels):
    img = img.cpu()
    img = img.reshape(-1,3,224,224)
    grid = make_grid(img, nrow=10, normalize=True, padding=0)
    #print(f"Test: ../data/participants/{par}/04_Image_Reconstruction/{task}/generated_img/epoch_{epoch}.png")
    save_image(grid,f"../data/participants/{par}/04_Image_Reconstruction/{task}/generated_img/epoch_{epoch}.png".format(par=par,task=task, epoch=epoch))
    fig, ax = plt.subplots(figsize=(20,100))
    ax.imshow(grid.permute(1, 2, 0).data)
    ax.axis('off')

def rearrange(eeg_latent, labels):
    labels_sorted = torch.argsort(labels)

    eeg_latent_ = eeg_latent[labels_sorted]
    labels_ = labels[labels_sorted]
    return eeg_latent_, labels_


def save_gen_img(imgs, labels, name):
    labels = labels.cpu()
    try:
        os.makedirs('../data/participants/{par}/04_Image_Reconstruction/{task}/FID/{name}/'.format(par=par,task=task,name=name))
    except:
        pass
    np.save('../data/participants/{par}/04_Image_Reconstruction/{task}/FID/{name}/generated_labels'.format(par=par,task=task,name=name), labels)
    for i in range(len(imgs)):
        img = imgs[i]
        img = img.cpu()
        img = img.reshape(-1,3,224,224)
        grid = make_grid(img, nrow=1, normalize=True, padding=0)
        save_image(grid,"../data/participants/{par}/04_Image_Reconstruction/{task}/FID/{name}/image_{i}.png".format(par=par,task=task,name=name,i=i))    

def save_class_desicion(d_class_decition, labels, name):
    labels = labels.cpu()
    d_classify_loss = d_classify_criterion(     d_class_decition.to(device)  ,    labels.to(device)        )
    try:
        os.makedirs('../data/participants/{par}/04_Image_Reconstruction/{task}/InceptionAccuracy/{name}/'.format(par=par,task=task,name=name))
    except:
        pass
    print(labels.cpu(), d_class_decition.cpu().detach().numpy(), d_classify_loss.cpu().detach().numpy())
    np.save('../data/participants/{par}/04_Image_Reconstruction/{task}/InceptionAccuracy/{name}/labels'.format(par=par,task=task,name=name), labels.cpu())
    np.save('../data/participants/{par}/04_Image_Reconstruction/{task}/InceptionAccuracy/{name}/d_class_decition'.format(par=par,task=task,name=name), d_class_decition.cpu().detach().numpy())
    np.save('../data/participants/{par}/04_Image_Reconstruction/{task}/InceptionAccuracy/{name}/d_inception_acc'.format(par=par,task=task,name=name), d_classify_loss.cpu().detach().numpy())
