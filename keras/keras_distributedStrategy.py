import os
os.environ["CUDA_VISIBLE_DEVICES"]="4, 5, 6, 7"
# os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3, 4, 5, 6, 7"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def contours_2d(data, filename):

        cmap = plt.cm.jet

        r = 1
        c = len(data)

        if len(data) == 2:
            fig, axs = plt.subplots(r, c)
            for i in range(r):
                for j in range(c):

                    if j == 0:
                        im = axs[j].imshow(data['Truth'], cmap=cmap, vmax=np.amax(data['Truth']), vmin=np.amin(
                            data['Truth']), filterrad=8.0)
                        axs[j].axis('off')
                        #axs[j].title.set_text('Truth')
                    else:
                        im = axs[j].imshow(data['Predicted'], cmap=cmap, vmax=np.amax(data['Predicted']),
                                           vmin=np.amin(data['Predicted']), filterrad=8.0)
                        axs[j].axis('off')
                        #axs[j].title.set_text('Predicted')

                    # divider = make_axes_locatable(axs[j])
                    # cax = divider.append_axes('right', size='7%', pad=0.15)
                    # fig.colorbar(im, cax=cax, orientation='vertical', shrink=0.1)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
            fig.colorbar(im, cax=cbar_ax)
            plt.savefig(filename, bbox_inches = 'tight',  pad_inches = 0)
            plt.close()
        elif len(data) == 3:
            fig, axs = plt.subplots(r, c)
            for i in range(r):
                for j in range(c):

                    if j == 0:
                        im = axs[j].imshow(data['Truth'], cmap=cmap, vmax=np.amax(data['Truth']), vmin=np.amin(
                            data['Truth']))
                        axs[j].axis('off')
                        axs[j].title.set_text('Truth')
                    elif j == 1:
                        im = axs[j].imshow(data['Predicted'], cmap=cmap, vmax=np.amax(data['Predicted']),
                                           vmin=np.amin(data['Predicted']))
                        axs[j].axis('off')
                        axs[j].title.set_text('Predicted')
                    else:
                        im = axs[j].imshow(data['Error'], cmap=cmap, vmax=np.amax(data['Error']),
                                           vmin=np.amin(data['Error']))
                        axs[j].axis('off')
                        axs[j].title.set_text('Absolute Error')

                    divider = make_axes_locatable(axs[j])
                    cax = divider.append_axes('right', size='7%', pad=0.1)
                    fig.colorbar(im, cax=cax, orientation='vertical', shrink=0.1);

            fig.subplots_adjust(right=0.8)
            plt.savefig(filename)
            plt.close()

        elif len(data) == 4:
            fig, axs = plt.subplots(r, c)
            for i in range(r):
                for j in range(c):

                    if j == 0:
                        im = axs[j].imshow(data['Diffusivity'], cmap=cmap, vmax=np.amax(data['Diffusivity']), vmin=np.amin(
                            data['Diffusivity']))
                        axs[j].axis('off')
                        axs[j].title.set_text('Diffusivity')
                    elif j == 1:
                        im = axs[j].imshow(data['Truth'], cmap=cmap, vmax=np.amax(data['Truth']), vmin=np.amin(
                            data['Truth']))
                        axs[j].axis('off')
                        axs[j].title.set_text('Truth')
                    elif j == 2:
                        im = axs[j].imshow(data['Predicted'], cmap=cmap, vmax=np.amax(data['Predicted']),
                                           vmin=np.amin(data['Predicted']))
                        axs[j].axis('off')
                        axs[j].title.set_text('Predicted')
                    elif j == 3:
                        im = axs[j].imshow(data['Error'], cmap=cmap, vmax=np.amax(data['Error']),
                                           vmin=np.amin(data['Error']))
                        axs[j].axis('off')
                        axs[j].title.set_text('Absolute Error')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
            fig.colorbar(im, cax=cbar_ax)
            plt.savefig(filename)
            plt.close()


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
        self.dense_layer_size = 2 * dense_layer_base
        self.num_variables = num_variables

        self.encoder = self.encoder_network()
        self.decoder = self.decoder_network()

    def encoder_network(self):

        flow_size = self.resolution

        input = tf.keras.layers.Input(shape=(flow_size[0], flow_size[1], self.num_variables))

        x = tf.keras.layers.Conv2D(self.filters, activation="relu", kernel_size=3, padding="same")(input) 

        for j in range(self.image_compression):
            x = tf.keras.layers.MaxPooling2D()(x)
            self.filters = int(self.filters * 2)
            x = tf.keras.layers.Conv2D(self.filters, activation=self.activation, kernel_size=3, padding="same")(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.dense_layer_size, activation="tanh")(x)
        enc = tf.keras.layers.Dense(self.latent_size, activation="tanh")(x)
        return tf.keras.Model(inputs=input, outputs=enc)

    def decoder_network(self):

        flow_size = self.resolution
        filter_reshape = self.dense_layer_size/(int(flow_size[0]/16)*int(flow_size[1]/16))

        enc = tf.keras.layers.Input(shape=(self.latent_size,))

        x = tf.keras.layers.Dense(self.dense_layer_size, activation="tanh")(enc)
        x = tf.keras.layers.Reshape((int(flow_size[0]/2**self.image_compression),\
         int(flow_size[1]/2**self.image_compression), int(filter_reshape)))(x)

        for j in range(self.image_compression):
            x = tf.keras.layers.Conv2D(self.filters, activation=self.activation, kernel_size=3, padding="same")(x)
            x = tf.keras.layers.UpSampling2D()(x)
            self.filters = int(self.filters/2)

        x = tf.keras.layers.Conv2D(self.filters, activation="relu", kernel_size=3, padding="same")(x)  # 4x4x64
        output = tf.keras.layers.Conv2D(self.num_variables, kernel_size=3, padding="same")(x)

        return tf.keras.Model(inputs=enc, outputs=output)

    def call(self, inputs, training=None, mask=None):
        enc = self.encoder(inputs)
        output = self.decoder(enc)

        return output

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)



variable_name = 'condition'
model_path = "./models/"
data_path = "./data/"+variable_name+"_data/"
# block_directory = "./data/blocked_data/"
block_directory = "./data/blocked_data/"

domain_size = (1024, 1024)
subdomain_size = (64, 64)
num_conditions = 1
total_subdomains = int(domain_size[0]/subdomain_size[0]) * int(domain_size[1]/subdomain_size[1])
num_dom = (int(domain_size[0] / subdomain_size[0]), int(domain_size[1] / subdomain_size[1]))
num_neighbors = 5
latent_size = 11
resolution = domain_size

## Load and reshape train data
try:
    blocked_train_data = np.load(block_directory + "blocked_" + "train" + "_" + variable_name + ".npy")
except:
    print("No data in ", data_path, " Please set process_data flag to True.")
    
if len(blocked_train_data.shape) == 4:
    blocked_train_data = np.expand_dims(blocked_train_data, -1)
blocked_train_data = blocked_train_data.reshape(
    (blocked_train_data.shape[0] * total_subdomains, subdomain_size[0], subdomain_size[1], blocked_train_data.shape[-1]))

## Load and reshape test data
try:
    blocked_test_data = np.load(block_directory + "blocked_" + "test" + "_" + variable_name + ".npy")
except:
    print("No data in ", data_path, " Please set process_data flag to True.")
    
if len(blocked_test_data.shape) == 4:
    blocked_test_data = np.expand_dims(blocked_test_data, -1)
blocked_test_data = blocked_test_data.reshape(
    (blocked_test_data.shape[0] * total_subdomains, subdomain_size[0], subdomain_size[1], blocked_test_data.shape[-1]))

# Data Normalization
mx = blocked_train_data.max()
mn = blocked_train_data.min()
np.save(block_directory + variable_name + "_mx.npy", mx)
np.save(block_directory + variable_name + "_mn.npy", mn)

print("Train data max =%f, Train data min=%f" % (mx, mn))
print("Train data shape =", blocked_train_data.shape)

blocked_train_data = (blocked_train_data - mn) / (mx - mn)
blocked_test_data = (blocked_test_data - mn) / (mx - mn)


# mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:4", "GPU:5", "GPU:6", "GPU:7"])
mirrored_strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))
with mirrored_strategy.scope():
    model = SolutionGenerator2D(subdomain_size, latent_size, filters=8, activation="relu", \
        image_compression=4, num_variables=1, dense_layer_base=64)
    opt = tf.keras.optimizers.Adam(1e-3)
    loss="mse"
    model.compile(loss=loss, optimizer=opt)

# model = SolutionGenerator2D(subdomain_size, latent_size, filters=8, activation="relu", \
#     image_compression=4, num_variables=1, dense_layer_base=64)
# model.built=True
print(model.encoder.summary())
print(model.decoder.summary())

model_name = 'auto_encoder.hdf5'
checkpoint_filepath = model_path + '/'
if not os.path.exists(checkpoint_filepath):
    os.makedirs(checkpoint_filepath)

load_model = False
if load_model:
    model.built = True
    model.load_weights(checkpoint_filepath + model_name)
    print("Model loaded")

train_mode = True
if train_mode:
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath + model_name,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True, verbose=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                        patience=15, min_lr=1e-6, cooldown=15, min_delta=1e-9,
                                                        verbose=True)
    input_data = blocked_train_data
    ind = np.arange(input_data.shape[0])
    np.random.shuffle(ind)
    input_data = input_data[ind]
    output_data = input_data

    model.fit(input_data, output_data, batch_size=512//1, epochs=5000, verbose=True,
                    validation_split=0.2, callbacks=[model_checkpoint_callback, reduce_lr])
else:
        val_data = blocked_test_data


        tr_mx = np.load(block_directory+variable_name+"_mx.npy")
        tr_mn = np.load(block_directory+variable_name+"_mn.npy")
        print("Variable=%s, Max=%f, Min=%f"%(variable_name, tr_mx, tr_mn))

        # val_data_n = (val_data.copy() - tr_mn) / (tr_mx - tr_mn)
        val_data_n = val_data.copy()

        val_pred = model.predict(val_data_n.reshape((val_data.shape[0] * total_subdomains, subdomain_size[0],
                                                        subdomain_size[1], num_conditions)), batch_size=2048)

        val_pred = val_pred.reshape(val_data.shape[0], total_subdomains, subdomain_size[0],
                                    subdomain_size[1], num_conditions)

        val_pred = (val_pred) * (tr_mx - tr_mn) + tr_mn


        val_data = val_data.reshape(val_data.shape[0], total_subdomains, *subdomain_size, num_conditions)
        num_vars = num_conditions

        # for iv in range(val_data.shape[-1]):
        for iv in range(num_vars):
            data_pred = []
            data_true = []
            print('val_pred shape: ', val_pred.shape)
            for k in range(val_pred.shape[0]):
                data_pred.append(uncubify(val_pred[k, :, :, :, iv], domain_size))
                data_true.append(uncubify(val_data[k, :, :, :, iv], domain_size))

            data_pred = np.asarray(data_pred)
            data_true = np.asarray(data_true)
            # print(data_true.shape, data_pred.shape)
            # raise
            file_plot = "../plots/generative_results/"+flags["variable"]+"/variable_"+str(iv)+"/"
            if not os.path.exists(file_plot):
                os.makedirs(file_plot)
            # print('data_true.shape', data_true.shape)
            # raise

            for j in range(100):
                val_sample = j
                filename = file_plot + str(val_sample)

                data = {}
                data["Truth"] = data_true[val_sample]
                data["Predicted"] = data_pred[val_sample]
                data["Error"] = np.abs(data_true[val_sample] - data_pred[val_sample])
                print(j, np.mean(data["Error"]))
                contours_2d(data, filename)


