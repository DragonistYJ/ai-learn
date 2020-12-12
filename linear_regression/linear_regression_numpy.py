import numpy


def create_parameters(dimension=4):
    """
    随机选定参数
    :param dimension: 参数维度，需要加1维的bias
    :return: 参数
    """
    return numpy.random.randint(low=0, high=10, size=dimension + 1)


def create_data(parameters: numpy.ndarray, size=500, dimension=4):
    """
    创建数据
    :param parameters: 参数
    :param size: 数据集大小
    :param dimension: 数据维度
    :return:
    """
    x = numpy.random.randint(low=0, high=50, size=(size, dimension))
    x = numpy.insert(x, 0, 1, axis=1)
    x = numpy.transpose(x)
    y = numpy.matmul(parameters, x)
    return x, y


def linear_regression(x: numpy.ndarray, y: numpy.ndarray, learning_rate=0.0001, epoch=2000):
    dimension = x.shape[0] - 1
    size = x.shape[1]
    p = numpy.random.randint(low=0, high=10, size=dimension + 1)

    for step in range(epoch):
        tem_p = numpy.matmul(p, x)
        tem_p = tem_p - y
        tem_p = numpy.matmul(x, numpy.transpose(tem_p))
        tem_p = tem_p / size
        p = p - learning_rate * numpy.transpose(tem_p)
        if (step + 1) % 10000 == 0:
            print(step, p)


def normal_equation(x: numpy.ndarray, y: numpy.ndarray):
    temp = numpy.matmul(x, numpy.transpose(x))
    temp = numpy.linalg.pinv(temp)
    temp = numpy.matmul(temp, x)
    temp = numpy.matmul(temp, y)
    return temp


if __name__ == '__main__':
    parameters = create_parameters()
    x_train, y_train = create_data(parameters, size=10)
    # linear_regression(x_train, y_train, epoch=1000000)
    print(parameters)
    print(normal_equation(x_train, y_train))
