from model import SolutionGenerator2D
from optparse import OptionParser
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def contours_2d(data, filename):

        cmap = plt.cm.jet

        r = 1
        c = len(data)

        if len(data) == 3:
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



if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m', '--mode', dest='mode',
                          help='mode: training or testing')
    parser.add_option('-v', '--variable', dest='variable',
                          help='variable: solution, condition')
    parser.add_option('-g', '--gpu', dest='gpu',
                          help='gpu: #')
    parser.add_option('-l', '--load_model', dest='load model',
                          help='load model: True or False')
    parser.add_option('-p', '--process_data', dest='process data',
                      help='process data: True or False')

    (options, args) = parser.parse_args()

    flags = {}
    for j in options.__dict__:
        flags[j] = options.__dict__[j]


    if flags["gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(flags["gpu"])
        print("Using gpu:", str(flags["gpu"]))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("Using gpu: 0")


    mode = flags["mode"]

    if flags["load model"]:
        load_model = eval(flags["load model"])
    else:
        load_model = False

    if flags["process data"]:
        process_data = eval(flags["process data"])
    else:
        process_data = False

    model_path = "./model/"
    data_path = "./data/"

    resolution = (160, 160)
    latent_size = 100

    model = SolutionGenerator2D(resolution, latent_size, filters=64, activation="relu", image_compression=4,
                     num_variables=1, dense_layer_base=64)

    model.encoder.summary()
    model.decoder.summary()



    def train(train_data, model, model_path, load_model, val_data=None, loss="mse"):

        input_data = train_data[0]
        output_data = train_data[1]

        model_name = 'auto_encoder.hdf5'
        checkpoint_filepath = model_path + '/'
        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath + model_name,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True, verbose=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                         patience=15, min_lr=1e-6, cooldown=15, min_delta=1e-9,
                                                         verbose=True)

        ind = np.arange(input_data.shape[0])
        np.random.shuffle(ind)
        input_data = input_data[ind]
        output_data = output_data[ind]
        # print('load_model: ', load_model);raise
        if load_model:
            model.built = True
            model.load_weights(checkpoint_filepath + model_name)
            print("Model loaded")

        if loss == "custom_loss":
            def custom_loss(y_true, y_pred):
                l2_true = tf.sqrt(tf.reduce_sum(tf.square(y_true), axis=(1)))
                l2_error = tf.sqrt(tf.reduce_sum(tf.square(y_true-y_pred), axis=(1)))
                loss = tf.reduce_mean(l2_error/l2_true)
                return loss
            loss = custom_loss

        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.lr
            return lr

        opt = tf.keras.optimizers.Adam(1e-3)
        lr_metric = get_lr_metric(opt)

        
        model.compile(loss=loss, optimizer=opt, metrics=lr_metric)


        if val_data is None:
            # print('No val data, batch_size', 51)
            model.fit(input_data, output_data, batch_size=64, epochs=50000, verbose=True,
                            validation_split=0.2, callbacks=[model_checkpoint_callback, reduce_lr])
        else:
            print('val_data available!')
            model.fit(input_data, output_data, batch_size=512*5, epochs=50000, verbose=True,
                            validation_data=val_data, callbacks=[model_checkpoint_callback, reduce_lr])


    data_directory = "./data/"
    if mode == "training":
        try:
            x_train = np.load(data_directory + "x_train" + ".npy").astype(np.float32)
            y_train = np.load(data_directory + "y_train" + ".npy").astype(np.float32)
        except:
            print("No data in ", data_path, " Please set process_data flag to True.")



        # print("Training data max=%f, Training data min=%f" % (train_data.max(), train_data.min()))
        # print("Training data shape=", train_data.shape)

        np.save(data_directory + "con_mx.npy", x_train.max())
        np.save(data_directory + "con_mn.npy", x_train.min())
        np.save(data_directory + "sol_mx.npy", y_train.max())
        np.save(data_directory + "sol_mn.npy", y_train.min())

        # Normalize training data
        x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())

        y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())
        x_train = np.expand_dims(x_train, -1)
        y_train = np.expand_dims(y_train, -1)
        # print('x_train.shape: ', x_train.shape)
        # raise
        # im = plt.imshow(x_train[0,:,:,0])
        # plt.colorbar(im)
        # plt.show()
        # raise
        train_data = (x_train, y_train)

        train(train_data, model, model_path, load_model, val_data=None, loss="mse")

    elif mode == "testing":
        if load_model:
            checkpoint_filepath = model_path + '/'
            model_name = 'auto_encoder.hdf5'
            model.load_weights(checkpoint_filepath + model_name)
            print("Model weights loaded")

        try:
            # val_data = np.load(block_directory + "blocked_" + "test" + "_" + flags["variable"] + ".npy")
            x_train = np.load(data_directory + "x_test" + ".npy").astype(np.float32)
            val_data = x_train
            y_train = np.load(data_directory + "y_test" + ".npy").astype(np.float32)
        except:
            print("No data in ", data_path, " Please set process_data flag to True.")

        pwr_tr_mx = np.load(data_directory + "con_mx.npy")
        pwr_tr_mn = np.load(data_directory + "con_mn.npy")
        print("Variable=%s, Max=%f, Min=%f"%("Power map:", pwr_tr_mx, pwr_tr_mn))

        val_data_n = (val_data.copy() - pwr_tr_mn) / (pwr_tr_mx - pwr_tr_mn)
        val_pred = model.predict(val_data_n, batch_size=2048)


        T_tr_mx = np.load(data_directory + "sol_mx.npy")
        T_tr_mn = np.load(data_directory + "sol_mn.npy")

        val_pred = (val_pred) * (T_tr_mx - T_tr_mn) + T_tr_mn


        # for iv in range(val_data.shape[-1]):
        # for iv in range(num_vars):
        data_pred = []
        data_true = []
        print('val_pred shape: ', val_pred.shape)
        for k in range(val_pred.shape[0]):
            data_pred.append(val_pred[k,:,:,0])
            data_true.append(y_train[k])
        data_pred = np.asarray(data_pred)
        data_true = np.asarray(data_true)

        file_plot = "./plots/"
        if not os.path.exists(file_plot):
            os.makedirs(file_plot)

        print('data_pred shape: ', data_pred.shape)
        print('data_true shape: ', data_true.shape)
        # raise
        for j in range(10):
            val_sample = j
            filename = file_plot + str(val_sample)

            data = {}
            data["Truth"] = data_true[val_sample]
            data["Predicted"] = data_pred[val_sample]
            data["Error"] = np.abs(data_true[val_sample] - data_pred[val_sample])
            print(j, np.mean(data["Error"]))
            contours_2d(data, filename)
