import numpy
import tensorflow.keras as keras


class LinearRegression(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dense1 = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        return self.__dense1(inputs)

    def get_config(self):
        pass

    def compile(self,
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.MeanSquaredError(),
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                run_eagerly=None,
                **kwargs):
        if metrics is None:
            metrics = [keras.metrics.MeanSquaredError()]
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, **kwargs)


def create_data(parameters=None, size=40000, dimension=10):
    if parameters is None:
        parameters = numpy.random.randint(low=0, high=10, size=dimension + 1)
    x = numpy.random.randint(low=0, high=50, size=(size, dimension))
    y = numpy.matmul(parameters, numpy.transpose(numpy.insert(x, 0, 1, axis=1)))
    return x, y, parameters


if __name__ == '__main__':
    linear_regression = LinearRegression()
    linear_regression.compile()
    x_train, y_train, parameter = create_data(parameters=[7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    linear_regression.fit(x=x_train, y=y_train, epochs=30)
    print(linear_regression.predict([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
