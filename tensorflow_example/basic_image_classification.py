import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# 本指南使用Fashion MNIST数据集，该数据集包含10个类别的70, 000个灰度图像。这些图像以低分辨率（28x28像素）展示了单件衣物
# Fashion MNIST旨在临时替代经典MNIST数据集，后者常被用作计算机视觉机器学习程序的“Hello, World”。MNIST数据集包含手写数字（0、1、2等）的图像，其格式与您将使用的衣物图像的格式相同。
# 本指南使用Fashion MNIST来实现多样化，因为它比常规MNIST更具挑战性。这两个数据集都相对较小，都用于验证某个算法是否按预期工作。对于代码的测试和调试，它们都是很好的起点。
# 在本指南中，我们使用60, 000个图像来训练网络，使用10, 000个图像来评估网络学习对图像分类的准确率。
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# 加载数据集会返回四个NumPy数组：
# train_images和train_labels数组是训练集，即模型用于学习的数据。
# test_images和test_labels数组会被用来对模型进行测试。
# 图像是28x28的NumPy数组，像素值介于0到255之间。标签是整数数组，介于0到9之间。
# 每个图像都会被映射到一个标签。由于数据集不包括类名称，请将它们存储在下方，供稍后绘制图像时使用：
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 在训练网络之前，必须对数据进行预处理。如果您检查训练集中的第一个图像，您会看到像素值处于0到255之间：
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
# 是否有网格线
plt.grid(False)
plt.show()

# 将这些值缩小至0到1之间，然后将其馈送到神经网络模型。为此，请将这些值除以255。请务必以相同的方式对训练集和测试集进行预处理
# 为了保证精度，需要将uint8转换成double，double类型的灰度数据默认范围在[0, 255]
train_images = train_images / 255.0
test_images = test_images / 255.0

# 为了验证数据的格式是否正确，以及您是否已准备好构建和训练网络，让我们显示训练集中的前25个图像，并在每个图像下方显示类名称
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 神经网络的基本组成部分是层。层会从向其馈送的数据中提取表示形式。希望这些表示形式有助于解决手头上的问题。
# 大多数深度学习都包括将简单的层链接在一起。大多数层（如tf.keras.layers.Dense）都具有在训练期间才会学习的参数。
# 该网络的第一层tf.keras.layers.Flatten将图像格式从二维数组（28x28像素）转换成一维数组（28x28 = 784像素）。将该层视为图像中未堆叠的像素行并将其排列起来。该层没有要学习的参数，它只会重新格式化数据。
# 展平像素后，网络会包括两个tf.keras.layers.Dense层的序列。它们是密集连接或全连接神经层。
# 第一个Dense层有128个节点（或神经元）。第二个（也是最后一个）层会返回一个长度为10的logits数组。每个节点都包含一个得分，用来表示当前图像属于10个类中的哪一类。
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# 在准备对模型进行训练之前，还需要再对其进行一些设置。以下内容是在模型的编译步骤中添加的：
#
# - 损失函数 - 用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
# - 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
# - 指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# 训练神经网络模型需要执行以下步骤：
# 1.将训练数据馈送给模型。在本例中，训练数据位于train_images和train_labels数组中。
# 2.模型学习将图像和标签关联起来。
# 3.要求模型对测试集（在本例中为test_images数组）进行预测。
# 4.验证预测是否与test_labels数组中的标签相匹配。

# 要开始训练，请调用model.fit方法，这样命名是因为该方法会将模型与训练数据进行“拟合”
# 在模型训练期间，会显示损失和准确率指标。此模型在训练数据上的准确率达到了0.91（或91 %）左右
model.fit(train_images, train_labels, epochs=10)

# 接下来，比较模型在测试数据集上的表现
# 结果表明，模型在测试数据集上的准确率略低于训练数据集。训练准确率和测试准确率之间的差距代表过拟合。过拟合是指机器学习模型在新的、以前未曾见过的输入上的表现不如在训练数据上的表现。过拟合的模型会“记住”训练数据集中的噪声和细节，从而对模型在新数据上的表现产生负面影响。
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 在模型经过训练后，您可以使用它对一些图像进行预测。模型具有线性输出，即logits。您可以附加一个softmax层，将logits转换成更容易理解的概率
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(np.argmax(predictions[0]), test_labels[0])
print(class_names[np.argmax(predictions[0])])


# 您可以将其绘制成图表，看看模型对于全部10个类的预测
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 在模型经过训练后，您可以使用它对一些图像进行预测。
# 我们来看看第0个图像、预测结果和预测数组。正确的预测标签为蓝色，错误的预测标签为红色。数字表示预测标签的百分比（总计为100）
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 使用训练好的模型
# tf.keras模型经过了优化，可同时对一个批或一组样本进行预测。因此，即便您只使用一个图像，您也需要将其添加到列表中
img = test_images[1]
print(img.shape)

img = np.expand_dims(img, 0)

predictions_single = probability_model.predict(img)
print(predictions_single)
print(np.argmax(predictions_single))

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
