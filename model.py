import torch
from torch import nn
import copy
from torchvision.io import read_image
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class CoRDataset(Dataset):
    def __init__(self, path):
        self.imgs_path = path
        self.class_map = {}
        file_list = glob.glob(self.imgs_path + "*")
        self.data=[]
        self.img =[]
        for i, class_path in enumerate(file_list):
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
                img=plt.imread(img_path)
                img =  img/np.amax(img)
                R, G, B = img[::4,::4,0], img[::4,::4,1], img[::4,::4,2]
                img = 0.2989 * R + 0.5870 * G + 0.1140 * B # On passe l'image en niveaux de gris pour que ça soit plus rapide
                self.img.append(img)
            self.class_map[class_name] = i
        self.img_dim = (56, 56)    
    
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(self.img[idx])
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id.float()
    

def Print_loss_accuracy(nepoch, tloss, vloss, accuracy, best_tloss, best_vloss, best_accuracy):
    print ("{:<6} {:<15} {:<17} {:<15} {:<20} {:<22} {:<15}".format(nepoch, tloss, vloss, accuracy, best_tloss, best_vloss, best_accuracy))

def Learning(nepoch, model, crit, optim, batchsize, trainingloader, validationloader):
    #writer = SummaryWriter(optim)
    best_tloss = 100.
    best_vloss = 100.
    best_accuracy = 0.
    Print_loss_accuracy('Epoch', 'training loss', 'validation loss', 'accuracy', 'best train loss', 'best validation loss', 'best accuracy')
    for nepoch in range(nepoch):
        tloss = 0.
        vloss = 0.
        correct_test = 0
        model.train()
        for images, labels in trainingloader:
            optim.zero_grad() 
            predicted = model(images.view(1, -1))
            loss = crit(predicted.squeeze(), labels.squeeze())
            loss.backward()
            optim.step()
            tloss += loss.item() * images.size(0)
        tloss /= len(trainingloader.sampler)
        model.eval()
        for images, labels in validationloader:
            predicted = model(images.view(1,-1)) 
            loss = crit(predicted.squeeze(), labels.squeeze())
            correct_test += np.sum(torch.squeeze(predicted).round().detach().numpy() == labels.squeeze().detach().numpy())
            vloss += loss.item() * images.size(0)
        vloss /= len(validationloader.sampler)
        accuracy = 100 * correct_test/len(validationloader)
        #writer.add_scalar('Accuracy', accuracy, nepoch)
        #writer.add_scalar('Loss/test', tloss, nepoch)
        #writer.add_scalar('Loss/validation', vloss, nepoch)

        if accuracy >= best_accuracy:
            torch.save(model, "best_model.pth")
            best_accuracy=accuracy
        if vloss <= best_vloss:
            best_vloss = vloss
        if tloss <= best_tloss:
            best_tloss = tloss
        Print_loss_accuracy(nepoch+1, 
                            np.round(tloss,8), 
                            np.round(vloss,8), 
                            np.round(accuracy,8), 
                            np.round(best_tloss,8), 
                            np.round(best_vloss,8), 
                            np.round(best_accuracy,8))
        
def Testmodel(modelfile,crit, testloader):
    model = torch.load(modelfile)
    model.eval()
    plt.figure(dpi=300)
    ct=1
    for imgs, labels in testloader:
        image=imgs[0]
        plt.subplot(1, len(test_loader.sampler),ct)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        predicted = model(imgs.view(1,-1)) 
        test_loss = crit(predicted.squeeze(), labels.squeeze())
        plt.title('True label : {} \n Predicted label : {} \n Test loss : {}'.format(labels.squeeze().detach().numpy(), 
                                                                       torch.squeeze(predicted).round().detach().numpy(),
                                                                                    np.round(test_loss.item(), 2)),
                  fontsize=6)
        ct+=1

BATCH_SIZE = 200
if __name__ == "__main__":
    training_set = CoRDataset("data/training/")
    training_loader = DataLoader(training_set, shuffle=True)
    validation_set = CoRDataset("data/validation/")
    validation_loader = DataLoader(validation_set, shuffle=False)
    test_set = CoRDataset("data/test/")
    test_loader = DataLoader(test_set, shuffle=False)

pokemonmodel = torch.nn.Sequential(torch.nn.Linear(56*56,20),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(20,10),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(10,1),
                                       )
criterion = torch.nn.CrossEntropyLoss()
#for val in [0, 10**(-5), 10**(-2), 10**0, 10**2]:
optimizer = torch.optim.SGD(pokemonmodel.parameters(), lr = 0.001, momentum=0.9, weight_decay = 10**(-5))
Learning(100, pokemonmodel, criterion, optimizer, BATCH_SIZE, training_loader, validation_loader)
Testmodel("best_model.pth", criterion, test_loader)