import torch
import numpy as np
from model import Net
from torch.autograd import Variable
import tkinter
from tkinter import filedialog, Label, Button
import os
from PIL import Image, ImageTk

img_png = None


def predict(img_root, model) -> int:
    img = Image.open(img_root).resize((28, 28))
    img = img.convert('L')
    img_data = np.array(img)
    img_data = torch.from_numpy(img_data).float()
    img_data = img_data.view(1, 1, 28, 28)
    output = model(Variable(img_data).cuda())
    _, pred = torch.max(output, 1)

    return pred.cpu().item()


def main():
    model = Net()
    # 加载模型
    model.load_state_dict(torch.load('model.pth'))

    if torch.cuda.is_available():
        # 使用GPU
        model.cuda()
    model.eval()
    predict('./test.jpeg', model)
    window = tkinter.Tk()
    window.geometry("350x420")
    window.title("周厚溧的手写数字识别")
    lbl = Label(window, text="请选择需要识别的图片")
    lbl.pack(side="top", expand="yes", fill="x", padx="10", pady="10", ipadx="5", ipady="5")
    global img_png
    label_img = Label(window, image=img_png)
    label_img.pack(expand="yes", fill="both", padx="10", pady="10", ipadx="5", ipady="5")

    def clicked(cnn):
        file = filedialog.askopenfilename(initialdir=os.path.dirname(__file__))
        global img_png
        img_png = ImageTk.PhotoImage(Image.open(file).resize((200, 200)))
        label_img.configure(image=img_png)
        lbl.configure(text='预测结果为：%d' % predict(file, cnn))
        # print(file)
        # print(predict(file, cnn))

    btn = Button(window, text="选择图片", command=lambda: clicked(model))
    btn.pack(side="bottom", expand="yes", fill="x", padx="10", pady="10", ipadx="5", ipady="5")
    window.mainloop()


if __name__ == "__main__":
    main()
