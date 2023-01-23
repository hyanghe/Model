import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
from tqdm import tqdm
import tensorflow as tf
from custom_callback import CustomReduceLRoP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import re

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

def save_model_as_min_validation(model, checkpoint_dir, run_number):
    filename = "model={}_run={:02d}-min_val_loss.pt".format(model.name, run_number)
    file_path = os.path.join(checkpoint_dir, filename)
    model.save_weights(file_path)

def get_lowest_val_loss_checkpoint(model_name,
                                   checkpoint_dir,
                                   run_number):
    files = os.listdir(checkpoint_dir)

    # Get files only for relevant model and run
    file_format = "model={}_run={:02d}-min_val_loss.pt.index".format(model_name, run_number)
    pattern = re.compile(file_format)
    relevant_files = []
    for f in files:
        if pattern.match(f):
            relevant_files.append(f)
    if len(relevant_files) == 0:
        raise ValueError("No min validation loss checkpoint found for model {} and run {}".format(model_name.name,
                                                                                                  run_number))
    state_dict_filename = relevant_files[0].split(".index")[0]  # remove index postfix
    state_dict_path = os.path.join(checkpoint_dir, state_dict_filename)
    return state_dict_path

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
model_path = "./models_tf/"
data_path = "./data/"+variable_name+"_data/"
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

model = SolutionGenerator2D(subdomain_size, latent_size, filters=8, activation="relu", \
    image_compression=4, num_variables=1, dense_layer_base=64)
model.built=True
print(model.encoder.summary())
print(model.decoder.summary())


output_dir = "output_" + variable_name + "_darcy_demo"
checkpoint_dir = os.path.join(os.path.join(output_dir), "checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
run_number = 1

load_model = True
if load_model:
    state_dict_path = get_lowest_val_loss_checkpoint(model.name, checkpoint_dir, run_number)
    model.load_weights(state_dict_path)
    print("Model loaded")


batch_size = 512

train_mode = False
if train_mode:
    input_data = blocked_train_data
    # input_data = blocked_train_data[:576000//2]

    ind = np.arange(input_data.shape[0])
    np.random.shuffle(ind)
    input_data = input_data[ind]
    output_data = input_data

    opt = tf.keras.optimizers.Adam(1e-3)
    loss="mse"

    ## Keras training
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath + model_name,
    #     save_weights_only=True,
    #     monitor='val_loss',
    #     mode='min',
    #     save_best_only=True, verbose=True)

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
    #                                                     patience=15, min_lr=1e-6, cooldown=15, min_delta=1e-9,
    #                                                     verbose=True)
    # model.compile(loss=loss, optimizer=opt)
    # model.fit(input_data, output_data, batch_size=512, epochs=5000, verbose=True,
    #                 validation_split=0.2, callbacks=[model_checkpoint_callback, reduce_lr])
    ## Keras training

    ## TF training
    # Prepare the training dataset.
    train_ratio = 0.8
    train_idx = np.random.choice(blocked_train_data.shape[0],\
         size=int(blocked_train_data.shape[0]*train_ratio),\
        replace=False)
    val_idx = np.setxor1d(np.arange(blocked_train_data.shape[0]), train_idx)

    x_train, y_train = blocked_train_data[train_idx], blocked_train_data[train_idx]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    x_val, y_val = blocked_train_data[val_idx], blocked_train_data[val_idx]
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size*4)

    loss_fn = tf.keras.losses.MeanSquaredError()

    starter_learning_rate = 1e-3
    end_learning_rate = 1e-6
    import pdb; pdb.set_trace()
    decay_steps = 1500*len(x_train)//batch_size
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.5)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    train_metric = tf.keras.metrics.MeanSquaredError("train_mse")
    val_metric = tf.keras.metrics.MeanSquaredError("val_mse")

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss_value = loss_fn(y, pred)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_metric.update_state(y, pred)
        return loss_value

    @tf.function
    def test_step(x, y):
        val_pred = model(x, training=False)
        val_metric.update_state(y, val_pred)

    epochs = 5000
    min_val_loss = np.inf
    ##      Add lr scheduler
    # reduce_rl_plateau = CustomReduceLRoP(
    #                         factor=0.3,
    #                         patience=15,
    #                         # patience=2,
    #                         min_lr=1e-6,
    #                         cooldown=15,
    #                         min_delta=1e-9,
    #                         verbose=1, 
    #                         optim_lr=optimizer.learning_rate, 
    #                         # mode='max', # For classification accuracy
    #                         mode='min', # For regression mse
    #                         reduce_lin=False)
##      Add lr scheduler
    num_train_batches = np.ceil(x_train.shape[0] / batch_size)
    progress_bar = tqdm(total=epochs * num_train_batches)
    # reduce_rl_plateau.on_train_begin()
    for epoch in range(epochs):
        # print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            loss_value = train_step(x_batch_train, y_batch_train)
            progress_bar.update()
            # Log every 200 batches.
            # if step % 200 == 0:
            #     print(
            #         "Training loss (for one batch) at step %d: %.4f"
            #         % (step, float(loss_value))
            #     )
            #     print("Seen so far: %d samples" % ((step + 1) * batch_size))

        # Display metrics at the end of each epoch.
        train_loss = train_metric.result()
        # print("Training loss over epoch: %.4f" % (float(train_loss),))

        # Reset training metrics at the end of each epoch
        train_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            test_step(x_batch_val, y_batch_val)

        val_loss = val_metric.result()
        val_metric.reset_states()
        # print("Validation loss: %.4f" % (float(val_loss),))
        print("Time taken: %.2fs" % (time.time() - start_time))

        if val_loss < min_val_loss:
            print(f'val loss reduced from {min_val_loss} to {val_loss}, saving checkpoint')
            min_val_loss = val_loss
            save_model_as_min_validation(model, checkpoint_dir, run_number)



        # Progress bar
        cur_lr = optimizer._decayed_lr(tf.float32).numpy()
        progress_bar.update()
        progress_bar.set_postfix(
            epoch="{}".format(epoch),
            train_loss="{:.2e}".format(train_loss),
            val_loss="{:.2e}".format(val_loss),
            lr="{:.2e}".format(cur_lr)
                    )

        # reduce_rl_plateau.on_epoch_end(epoch, val_loss)

else:
        tr_mx = np.load(block_directory+variable_name+"_mx.npy")
        tr_mn = np.load(block_directory+variable_name+"_mn.npy")
        print("Variable=%s, Max=%f, Min=%f"%(variable_name, tr_mx, tr_mn))

        test_data = tf.data.Dataset.from_tensor_slices((blocked_test_data, blocked_test_data))
        test_data = test_data.batch(batch_size*4)

        GT = []
        Pred = []
        for i, X in enumerate(test_data):
            X = X[0]
            Y = model(X)
            Pred.extend(Y)
            GT.extend(X)
        Pred = np.asarray(Pred)
        GT = np.asarray(GT)

        Pred = Pred.reshape(Pred.shape[0]//total_subdomains, total_subdomains, subdomain_size[0],
                                    subdomain_size[1], num_conditions)
        GT = GT.reshape(GT.shape[0]//total_subdomains, total_subdomains, subdomain_size[0],
                                    subdomain_size[1], num_conditions)

        tr_mx = np.load(block_directory+variable_name+"_mx.npy")
        tr_mn = np.load(block_directory+variable_name+"_mn.npy")
        print("Variable=%s, Max=%f, Min=%f"%(variable_name, tr_mx, tr_mn))
        # Scale back
        Pred = (Pred) * (tr_mx - tr_mn) + tr_mn
        GT = GT * (tr_mx - tr_mn) + tr_mn

        for iv in range(num_conditions):
            data_pred = []
            data_true = []
            for k in range(Pred.shape[0]):
                data_pred.append(uncubify(Pred[k, :, :, :, iv], domain_size))
                data_true.append(uncubify(GT[k, :, :, :, iv], domain_size))
            data_pred = np.asarray(data_pred)
            data_true = np.asarray(data_true)

            file_plot = "./plots/generative_results_tf/"+variable_name+"/variable_"+str(iv)+"/"
            if not os.path.exists(file_plot):
                os.makedirs(file_plot)

            err_total = []
            for j in range(100):
                val_sample = j
                filename = file_plot + str(val_sample)        
                data = {}
                data["Truth"] = data_true[val_sample]
                data["Predicted"] = data_pred[val_sample]
                data["Error"] = np.abs(data_true[val_sample] - data_pred[val_sample])
                print(j, np.mean(data["Error"]))
                err_total.append(np.mean(data["Error"]))
                contours_2d(data, filename)
            print('MAE: ', sum(err_total)/len(err_total))


