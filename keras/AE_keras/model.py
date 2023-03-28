import tensorflow as tf


class SolutionGenerator2D(tf.keras.Model):
    def __init__(self, resolution, latent_size, filters=8, activation="relu", image_compression=4,
                 num_variables=1, dense_layer_base=64):
        super(SolutionGenerator2D, self).__init__()
        self.resolution = resolution
        self.latent_size = latent_size
        self.filters = filters
        self.activation = activation
        self.image_compression = image_compression
        self.dense_layer_base = dense_layer_base
        # self.dense_layer_size = 2 * dense_layer_base
        self.dense_layer_size = int(resolution[0]/2**self.image_compression)**2
        self.num_variables = num_variables
        self.kernel_size = 3
        self.encoder = self.encoder_network()
        self.decoder = self.decoder_network()

    def encoder_network(self):

        flow_size = self.resolution

        input = tf.keras.layers.Input(shape=(flow_size[0], flow_size[1], self.num_variables))
        # print('input shape: ', input.shape)
        # raise
        x = tf.keras.layers.Conv2D(self.filters, activation="relu", kernel_size=self.kernel_size, padding="same")(input) 

        for j in range(self.image_compression):
            x = tf.keras.layers.MaxPooling2D()(x)
            self.filters = int(self.filters * 2)
            x = tf.keras.layers.Conv2D(self.filters, activation=self.activation, kernel_size=self.kernel_size, padding="same")(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.dense_layer_size, activation="tanh")(x)
        enc = tf.keras.layers.Dense(self.latent_size, activation="tanh")(x)
        return tf.keras.Model(inputs=input, outputs=enc)

    def decoder_network(self):

        flow_size = self.resolution
        filter_reshape = self.dense_layer_size/(int(flow_size[0]/16)*int(flow_size[1]/16))

        enc = tf.keras.layers.Input(shape=(self.latent_size,))
        # print('enc shape: ', enc.shape)
        x = tf.keras.layers.Dense(self.dense_layer_size, activation="tanh")(enc)
        x = tf.keras.layers.Reshape((int(flow_size[0]/2**self.image_compression),\
         int(flow_size[1]/2**self.image_compression), int(filter_reshape)))(x)

        for j in range(self.image_compression):
            x = tf.keras.layers.Conv2D(self.filters, activation=self.activation, kernel_size=self.kernel_size, padding="same")(x)
            x = tf.keras.layers.UpSampling2D()(x)
            self.filters = int(self.filters/2)

        x = tf.keras.layers.Conv2D(self.filters, activation="relu", kernel_size=self.kernel_size, padding="same")(x)  # 4x4x64
        output = tf.keras.layers.Conv2D(self.num_variables, kernel_size=self.kernel_size, padding="same")(x)
        # print('output shape: ', output.shape)
        # raise
        return tf.keras.Model(inputs=enc, outputs=output)

    def call(self, inputs, training=None, mask=None):
        enc = self.encoder(inputs)
        output = self.decoder(enc)

        return output

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)