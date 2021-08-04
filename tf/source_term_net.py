# https://tfs.ansys.com:8443/tfs/ANSYS_Development/AML/_git/mlSolver?path=%2Fsrc%2Fsourceterm_encoder%2Fsourceterm_net.py&version=GBdevelop&_a=contents
# develop branch-->src/sourceterm_encoder/sourceterm_net.py
# Power map Data generation: E:\Pycharm_workspace\ML_Svr_PwmE\data
# Generation new data for MLSolver: F:\Fluent\ReadFluent.py
import tensorflow as tf
from tensorflow.keras import Model

class SourceTermNet(Model):

    def __init__(self, powermap_size, latent_size):
        super(SourceTermNet, self).__init__()
        self.powermap_size = powermap_size
        self.latent_size = latent_size
        #self.autoencoder = self.network()
        self.encoder = self.encoder_network()
        self.decoder = self.decoder_network()

    def ConvBlock(self, u, BN, activation, filters):
        for j in range(2):
            u = tf.keras.layers.Conv2D(int(filters), kernel_size=3, padding="same", activation=activation)(u)
            if BN:
                u = tf.keras.layers.BatchNormalization()(u)
        return u

    def encoder_network(self):
        filters = 32
        activation = "relu"
        BN = False

        Var = tf.keras.layers.Input(shape=(self.powermap_size[0] * self.powermap_size[1],), name='var')
        u = tf.keras.layers.Reshape((self.powermap_size[0], self.powermap_size[1], 1))(Var)  # 16*16
        for j in range(3):
            u = self.ConvBlock(u, BN, activation, filters)
            if j != 2:
                u = tf.keras.layers.MaxPooling2D()(u)
                filters *= 2
        u = tf.keras.layers.Flatten()(u)
        encoded = tf.keras.layers.Dense(int(self.latent_size * 8), activation='relu')(u)
        encoded = tf.keras.layers.Dense(int(self.latent_size), activation='tanh', name="latent_space")(encoded)

        model = tf.keras.Model(inputs=Var, outputs=encoded)

        return model

    def decoder_network(self):
        filters = 128
        activation = "relu"
        BN = False
        row = int(self.powermap_size[0] / 4)
        col = int(self.powermap_size[1] / 4)

        encoded = tf.keras.Input(shape=(self.latent_size,))
        x = tf.keras.layers.Dense(int(self.latent_size * 8), activation='relu')(encoded)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        u = tf.keras.layers.Reshape((row, col, int(512 / (row * col))))(x)
        for j in range(3):
            u = self.ConvBlock(u, BN, activation, filters)
            if j != 2:
                u = tf.keras.layers.UpSampling2D()(u)
                filters /= 2
        u = tf.keras.layers.Conv2D(1, kernel_size=3, padding="same")(u)
        u1 = tf.keras.layers.Flatten(name="out")(u)

        model = Model(inputs=[encoded], outputs=[u1])

        return model

    def network(self ):

        encoder = self.encoder_network()
        decoder = self.decoder_network()

        model = tf.keras.Sequential([
            encoder,
            decoder,
        ])
        return model

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, enc):
        decoded = self.decoder(enc)
        return decoded
