import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.transforms as T
import cv2
from PIL import Image
from tqdm import tqdm

from MyWork.public import self2self

# 图片加载
img = np.array(Image.open("D:\jupyter_notebook\paper\dataset\Kodak24\kodim01.png")).astype('float32')
img = np.resize(img,(512,512,3))
# img += np.random.normal(0,10,size=img.shape)

# 参数设置
##Enable GPU
USE_GPU = True

dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

learning_rate = 1e-4
model = self2self(3,0.3)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
w,h,c = img.shape
p=0.3
NPred=100
slice_avg = torch.tensor([1,3,512,512]).to(device)


# 训练迭代
def image_loader(image, device, p1, p2):
    """
        load image and returns cuda tensor
    """
    loader = T.Compose([
        T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),
        T.RandomVerticalFlip(torch.round(torch.tensor(p2))),
        T.ToTensor()])
    image = Image.fromarray(image.astype(np.uint8))
    image = loader(image).float()
    if not torch.is_tensor(image):
        image = torch.tensor(image)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.to(device)


pbar = tqdm(range(500000))

for itr in pbar:
    # 不知道这个采样是否正确，是不是需要在每一个通道都分别进行均匀采样？
    p_mtx = np.random.uniform(size=[img.shape[0], img.shape[1], img.shape[2]])
    mask = (p_mtx > p).astype(np.double)
    img_input = img

    y = img
    p1 = np.random.uniform(size=1)
    p2 = np.random.uniform(size=1)
    # 加载输入图片（根据概率进行翻转）
    img_input_tensor = image_loader(img_input, device, p1, p2)

    # 对原始图片进行相同操作（翻转）
    y = image_loader(y, device, p1, p2)

    # mask为伯努利采样结果
    mask = np.expand_dims(np.transpose(mask, [2, 0, 1]), 0)
    mask = torch.tensor(mask).to(device, dtype=torch.float32)

    # 网络推理
    model.train()
    img_input_tensor = img_input_tensor * mask
    output = model(img_input_tensor, mask)

    # 损失函数
    # loss = torch.sum((output+img_input_tensor-y)*(output+img_input_tensor-y)*(1-mask))/torch.sum(1-mask)
    loss = torch.sum((output - y) * (output - y) * (1 - mask)) / torch.sum(1 - mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_description("iteration {}, loss = {:.4f}".format(itr + 1, loss.item() * 100))

    if (itr + 1) % 1000 == 0:
        model.eval()
        sum_preds = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
        for j in range(NPred):
            p_mtx = np.random.uniform(size=img.shape)
            mask = (p_mtx > p).astype(np.double)
            img_input = img * mask
            img_input_tensor = image_loader(img_input, device, 0.1, 0.1)
            mask = np.expand_dims(np.transpose(mask, [2, 0, 1]), 0)
            mask = torch.tensor(mask).to(device, dtype=torch.float32)

            output_test = model(img_input_tensor, mask)
            sum_preds[:, :, :] += np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0]
        avg_preds = np.squeeze(
            np.uint8(np.clip((sum_preds - np.min(sum_preds)) / (np.max(sum_preds) - np.min(sum_preds)), 0, 1) * 255))
        write_img = Image.fromarray(avg_preds)
        write_img.save("./process_imgs/Self2self-" + str(itr + 1) + ".png")
        torch.save(model.state_dict(), './process_model/model-' + str(itr + 1))
