# functions for Mineral_AI, 2023-07-11
import os
import re
import keras
import lasio
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import tensorflow as tf
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from scipy.interpolate import interp2d, interpn, splrep, Rbf, RegularGridInterpolator, InterpolatedUnivariateSpline
from keras.layers import Dropout, Add, Lambda, Flatten, Dense, Reshape
from keras import backend as K
from keras.regularizers import l2
from keras.models import load_model
from keras.layers import Input, LeakyReLU
from keras.engine.training import Model
from keras.layers.convolutional import Conv1D, UpSampling1D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling1D
plt.rcParams.update({'font.size': 13})
# ------------------------------------------------------------------------------------------------------------------------------------
def build_model_1(input_layer, output_size, start_neurons, size=3, DropoutRatio=0.3):
    activation_fun = "sigmoid"
    conv1 = Conv1D(start_neurons * 1, size, activation=activation_fun, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling1D(2)(conv1)
    pool1 = Dropout(DropoutRatio)(pool1, training=True)
    conv2 = Conv1D(start_neurons * 2, size, activation=activation_fun, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling1D(2)(conv2)
    pool2 = Dropout(DropoutRatio)(pool2, training=True)
    conv3 = Conv1D(start_neurons * 4, size, activation="sigmoid", padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling1D(2)(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    convm = Conv1D(start_neurons * 8, size, activation=activation_fun, padding="same")(pool3)
    convm = residual_block(convm, start_neurons * 8)
    convm = residual_block(convm, start_neurons * 8, True)
    deconv3 = Conv1DTranspose(convm, start_neurons * 4, size, strides=2, padding="same")
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    uconv3 = Conv1D(start_neurons * 4, size, activation="sigmoid", padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)
    deconv2 = Conv1DTranspose(uconv3, start_neurons * 2, size, strides=2, padding="same")
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2, training=True)
    uconv2 = Conv1D(start_neurons * 2, size, activation=activation_fun, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)
    deconv1 = Conv1DTranspose(uconv2, start_neurons * 1, size, strides=2, padding="same")
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1, training=True)
    uconv1 = Conv1D(start_neurons * 1, size, activation=activation_fun, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    output_layer_noActi = Conv1D(1, 1, padding="same", activation=activation_fun)(uconv1)
    uconv0 = Flatten()(output_layer_noActi)
    uconv0 = Dense(128, activation=activation_fun, name='cnn_flatten')(uconv0)
    uconv0 = Dense(10, activation=activation_fun, name='cnn_flatten16')(uconv0)
    uconv0 = Dense(output_size, activation=activation_fun)(uconv0)
    model = Model(input_layer, uconv0)
    return model


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return x


def convolution_block(x, filters, size, strides=1, padding='same', activation=True):
    x = Conv1D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, 3)
    x = convolution_block(x, num_filters, 3, activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def Conv1DTranspose(input_tensor, filters, ksize, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(ksize, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


def _error(y_true, y_pred):
    return y_true - y_pred


def mse(y_true, y_pred):
    return K.mean(K.square(_error(y_true, y_pred)))


def rmse(y_true, y_pred):
    return K.sqrt(mse(y_true, y_pred))


def predict_dist(model, x_test, num_samples):
    preds = [model(x_test, training=True) for _ in range(num_samples)]
    return np.dstack(preds)


def predict_value(model, x_test, num_samples):
    pred_dist = predict_dist(model, x_test, num_samples)
    return pred_dist.mean(axis=2)


def train_XGBoost(model_name, X_train, y_train, X_valid, y_valid):
    model = xgb.XGBRegressor(n_estimators=1000, max_depth=11, eta=0.005, subsample=0.7, colsample_bytree=0.8,
                             n_jobs=100,
                             random_state=2021, tree_method='auto')
    model.fit(X_train, y_train, early_stopping_rounds=100, eval_set=[(X_valid, y_valid)], verbose=1)
    pickle.dump(model, open(model_name, 'wb'))


def plot_feature_stats(X, y, feature_names, facies_colors, facies_names):
    nan_idx = np.any(np.isnan(X), axis=1)
    X = X[np.logical_not(nan_idx), :]
    y = y[np.logical_not(nan_idx)]
    features = pd.DataFrame(X, columns=feature_names)
    labels = pd.DataFrame(y, columns=['Facies'])
    for f_idx, facies in enumerate(facies_names):
        labels[labels[:] == f_idx] = facies
    data = pd.concat((labels, features), axis=1)
    facies_color_map = {}
    for ind, label in enumerate(facies_names):
        facies_color_map[label] = facies_colors[ind]
    sns.pairplot(data, hue='Facies', palette=facies_color_map, hue_order=list(reversed(facies_names)))


def card2polwells(data_in, features_wells):
    data_polar = data_in
    fea_red = features_wells
    name_temp = features_wells
    for fea1 in features_wells:
        del fea_red[0]
        for fea2 in fea_red:
            x = data_in[fea1] / max(data_in[fea1])
            y = data_in[fea2] / max(data_in[fea2])
            data_polar[fea1 + '_' + fea2 + '_rho'] = np.sqrt(x ** 2 + y ** 2)
            data_polar[fea1 + '_' + fea2 + '_phi'] = np.arctan2(y, x)
    return data_polar


def print_confusion_matrix(confusion_matrix, class_names, normalize, figsize=(10, 7), fontsize=14):
    if normalize:
        row_sum = confusion_matrix.sum(axis=1)
        confusion_matrix = confusion_matrix.astype('float') / row_sum[:, np.newaxis]
        title = "Normalized confusion matrix"
    else:
        title = 'Confusion matrix, without normalization'
    df_cm = pd.DataFrame(confusion_matrix)
    df_cm = df_cm.rename(index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    plt.title(title)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, cmap='Blues')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha=plt.ylabel('True label'))
    plt.xlabel('Predicted label')
    return fig


def make_facies_log_plot(logs, facies_colors):
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
        facies_colors[0:len(facies_colors)], 'indexed')
    ztop = logs.Depth.min();
    zbot = logs.Depth.max()
    cluster = np.repeat(np.expand_dims(logs['Facies'].values, 1), 100, 1)
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im = ax[5].imshow(cluster, interpolation='none', aspect='auto',
                      cmap=cmap_facies, vmin=1, vmax=9)
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label((17 * ' ').join([' SS ', 'CSiS', 'FSiS',
                                    'SiSh', ' MS ', ' WS ', ' D  ',
                                    ' PS ', ' BS ']))
    cbar.set_ticks(range(0, 1));
    cbar.set_ticklabels('')
    for i in range(len(ax) - 1):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(), logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(), logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(), logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(), logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(), logs.PE.max())
    ax[5].set_xlabel('Facies')
    ax[1].set_yticklabels([]);
    ax[2].set_yticklabels([]);
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]);
    ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    f.suptitle('Well: %s' % logs.iloc[0]['Well Name'], fontsize=14, y=0.94)


def compare_facies_plot(logs, compadre, facies_colors):
    logs = logs.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(
        facies_colors[0:len(facies_colors)], 'indexed')
    ztop = logs.Depth.min();
    zbot = logs.Depth.max()
    cluster1 = np.repeat(np.expand_dims(logs['Facies'].values, 1), 100, 1)
    cluster2 = np.repeat(np.expand_dims(logs[compadre].values, 1), 100, 1)

    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(9, 12))
    ax[0].plot(logs.GR, logs.Depth, '-g')
    ax[1].plot(logs.ILD_log10, logs.Depth, '-')
    ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.5')
    ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
    ax[4].plot(logs.PE, logs.Depth, '-', color='black')
    im1 = ax[5].imshow(cluster1, interpolation='none', aspect='auto',
                       cmap=cmap_facies, vmin=1, vmax=9)
    im2 = ax[6].imshow(cluster2, interpolation='none', aspect='auto',
                       cmap=cmap_facies, vmin=1, vmax=9)
    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label((17 * ' ').join([' SS ', 'CSiS', 'FSiS',
                                    'SiSh', ' MS ', ' WS ', ' D ',
                                    ' PS ', ' BS ']))
    cbar.set_ticks(range(0, 1));
    cbar.set_ticklabels('')
    for i in range(len(ax) - 2):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
        ax[0].set_xlabel("GR")
    ax[0].set_xlim(logs.GR.min(), logs.GR.max())
    ax[1].set_xlabel("ILD_log10")
    ax[1].set_xlim(logs.ILD_log10.min(), logs.ILD_log10.max())
    ax[2].set_xlabel("DeltaPHI")
    ax[2].set_xlim(logs.DeltaPHI.min(), logs.DeltaPHI.max())
    ax[3].set_xlabel("PHIND")
    ax[3].set_xlim(logs.PHIND.min(), logs.PHIND.max())
    ax[4].set_xlabel("PE")
    ax[4].set_xlim(logs.PE.min(), logs.PE.max())
    ax[5].set_xlabel('Facies')
    ax[6].set_xlabel('Predictions')
    ax[1].set_yticklabels([])#loop
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_yticklabels([])
    ax[5].set_xticklabels([])
    ax[6].set_xticklabels([])
    ax[6].set_yticklabels([])
    f.suptitle('Well: %s' % logs.iloc[0]['Well Name'], fontsize=14, y=0.94)


def compare_minerals_plot(fn_save, logs, compadre, mineral_colors):
    logs = logs.sort_values(by='DEPT')
    cmap_mineral = colors.ListedColormap(mineral_colors[0:len(mineral_colors)], 'indexed')
    ztop = logs.DEPT.min();
    zbot = logs.DEPT.max()
    f, ax = plt.subplots(nrows=1, ncols=7, figsize=(19, 11))
    ax[0].plot(logs.GR, logs.DEPT, '-g', linewidth=0.7)
    ax[1].plot(logs.RD, logs.DEPT, '-', linewidth=0.7)
    ax[2].plot(logs.DTP, logs.DEPT, '-', color='0.5', linewidth=0.7)
    ax[3].plot(logs.DEN, logs.DEPT, '-', color='r', linewidth=0.7)
    ax[4].plot(logs.PE, logs.DEPT, '-', color='black', linewidth=0.7)
    ax[5].fill_betweenx(logs.DEPT, 0, logs.WQFM, color=mineral_colors[0])
    ax[5].fill_betweenx(logs.DEPT, logs.WQFM, logs.WQFM + logs.WCARB, color=mineral_colors[1])
    ax[5].fill_betweenx(logs.DEPT, logs.WQFM + logs.WCARB, logs.WQFM + logs.WCARB + logs.WCLAY, color=mineral_colors[2])
    ax[5].fill_betweenx(logs.DEPT, logs.WQFM + logs.WCARB + logs.WCLAY, logs.WQFM + logs.WCARB + logs.WCLAY + logs.WPYR,
                        color=mineral_colors[3])
    ax[6].fill_betweenx(logs.DEPT, 0, logs.WQFM_pre, color=mineral_colors[0])
    ax[6].fill_betweenx(logs.DEPT, logs.WQFM_pre, logs.WQFM_pre + logs.WCARB_pre, color=mineral_colors[1])
    ax[6].fill_betweenx(logs.DEPT, logs.WQFM_pre + logs.WCARB_pre, logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre,
                        color=mineral_colors[2])
    ax[6].fill_betweenx(logs.DEPT, logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre,
                        logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre + logs.WPYR_pre, color=mineral_colors[3])
    ax[6].legend(['QFM', 'CARB', 'CLAY', 'PYR'], loc='lower right', title='Mineralogy', bbox_to_anchor=(1.6, 0.01),
                 fancybox=True, shadow=True)

    for i in range(len(ax)):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    ax[0].set_xlabel("GR (API)")
    ax[0].set_xlim(logs.GR.min(), logs.GR.max())
    ax[1].set_xlabel("RD (ohm.m)")
    ax[1].set_xlim(logs.RD.min(), logs.RD.max())
    ax[2].set_xlabel("DTP (usec/m)")
    ax[2].set_xlim(logs.DTP.min(), logs.DTP.max())
    ax[3].set_xlabel("DEN (g/cc)") #unit
    ax[3].set_xlim(logs.DEN.min(), logs.DEN.max())
    ax[4].set_xlabel("PE (barns/e)")
    ax[4].set_xlim(logs.PE.min(), logs.PE.max())
    ax[5].set_xlabel('GCL')
    ax[5].set_xlim(0, 1)
    ax[6].set_xlabel('Pred_GCL')
    ax[6].set_xlim(0, 1)
    ax[1].set_yticklabels([]);
    ax[2].set_yticklabels([]);
    ax[3].set_yticklabels([]);
    ax[4].set_yticklabels([]);
    ax[5].set_yticklabels([]);
    ax[6].set_yticklabels([]);
    f.suptitle('Well: %s' % str(logs.iloc[0]['ID']), fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(fn_save, dpi=350)


def compare_minerals_plot_0(fn_save, logs, compadre, mineral_colors):
    logs = logs.sort_values(by='DEPT')
    cmap_mineral = colors.ListedColormap(mineral_colors[0:len(mineral_colors)], 'indexed')
    ztop = logs.DEPT.min();
    zbot = logs.DEPT.max()

    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(21, 11), constrained_layout=True)
    ax[0].plot(logs.GR, logs.DEPT, '-g', linewidth=0.7)
    ax[1].plot(logs.RD, logs.DEPT, '-', linewidth=0.7)
    ax[2].plot(logs.DTP, logs.DEPT, '-', color='0.5', linewidth=0.7)
    ax[3].plot(logs.DEN, logs.DEPT, '-', color='r', linewidth=0.7)
    ax[4].plot(logs.PE, logs.DEPT, '-', color='black', linewidth=0.7)
    ax[5].fill_betweenx(logs.DEPT, 0, logs.WQFM_pre, color=mineral_colors[0])
    ax[5].fill_betweenx(logs.DEPT, logs.WQFM_pre, logs.WQFM_pre + logs.WCARB_pre, color=mineral_colors[1])
    ax[5].fill_betweenx(logs.DEPT, logs.WQFM_pre + logs.WCARB_pre, logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre,
                        color=mineral_colors[2])
    ax[5].fill_betweenx(logs.DEPT, logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre,
                        logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre + logs.WPYR_pre, color=mineral_colors[3])
    ax[5].legend(['QFM', 'CARB', 'CLAY', 'PYR'], loc='lower right', title='Mineralogy', bbox_to_anchor=(1.6, 0.01),
                 fancybox=True, shadow=True)

    for i in range(len(ax)):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    ax[0].set_xlabel("GR (API)")
    ax[0].set_xlim(logs.GR.min(), logs.GR.max())
    ax[1].set_xlabel("RD (ohm.m)")
    ax[1].set_xlim(logs.RD.min(), logs.RD.max())
    ax[2].set_xlabel("DTP (usec/m)")
    ax[2].set_xlim(logs.DTP.min(), logs.DTP.max())
    ax[3].set_xlabel("DEN (g/cc)")
    ax[3].set_xlim(logs.DEN.min(), logs.DEN.max())
    ax[4].set_xlabel("PE (barns/e)")
    ax[4].set_xlim(logs.PE.min(), logs.PE.max())
    ax[5].set_xlabel('Pred_GCL')
    ax[5].set_xlim(0, 1)
    ax[0].set_ylabel('Depth (m)')
    ax[1].set_yticklabels([]);
    ax[2].set_yticklabels([]);
    ax[3].set_yticklabels([]);
    ax[4].set_yticklabels([]);
    ax[5].set_yticklabels([])
    f.suptitle('Well: %s' % str(logs.iloc[0]['ID']), fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(fn_save, dpi=300)


def compare_minerals_plot_1(fn_save, logs, compadre, mineral_colors, core_GCL):
    logs = logs.sort_values(by='DEPT')
    cmap_mineral = colors.ListedColormap(mineral_colors[0:len(mineral_colors)], 'indexed')
    core_temp = core_GCL[core_GCL['License'] == logs.ID[0]]

    ztop = logs.DEPT.min();
    zbot = logs.DEPT.max()

    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(19, 11))
    ax[0].plot(logs.WQFM_pre, logs.DEPT, '-', color=mineral_colors[0], linewidth=0.8, zorder=1)
    ax[0].scatter(core_temp['WQFM_wt%'] / 100, core_temp['Depth_m'], marker='.', c='mediumblue', s=7, zorder=2);
    ax[1].plot(logs.WCARB_pre, logs.DEPT, '-', color=mineral_colors[1], linewidth=0.8, zorder=1)
    ax[1].scatter(core_temp['WCARBONATE_wt%'] / 100, core_temp['Depth_m'], marker='.', c='mediumblue', s=7, zorder=2);
    ax[2].plot(logs.WCLAY_pre, logs.DEPT, '-', color=mineral_colors[2], linewidth=0.8, zorder=1)
    ax[2].scatter(core_temp['WCLAY_wt%'] / 100, core_temp['Depth_m'], marker='.', c='mediumblue', s=7, zorder=2);
    ax[3].plot(logs.WPYR_pre, logs.DEPT, '-', color=mineral_colors[3], linewidth=0.8, zorder=1)
    ax[3].scatter(core_temp['WPYRITE_wt%'] / 100, core_temp['Depth_m'], marker='.', c='mediumblue', s=7, zorder=2);

    ax[4].fill_betweenx(logs.DEPT, 0, logs.WQFM_pre, color=mineral_colors[0])
    ax[4].fill_betweenx(logs.DEPT, logs.WQFM_pre, logs.WCLAY_pre + logs.WCARB_pre, color=mineral_colors[1])
    ax[4].fill_betweenx(logs.DEPT, logs.WQFM_pre + logs.WCARB_pre, logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre,
                        color=mineral_colors[2])
    ax[4].fill_betweenx(logs.DEPT, logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre,
                        logs.WQFM_pre + logs.WCARB_pre + logs.WCLAY_pre + logs.WPYR_pre, color=mineral_colors[3])
    ax[4].legend(['QFM', 'CARB', 'CLAY', 'PYR'], loc='lower right', title='Mineralogy', bbox_to_anchor=(1.5, 0.01),
                 fancybox=True, shadow=True)

    for i in range(len(ax)):
        ax[i].set_ylim(ztop, zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    ax[0].set_xlabel("QFM (g/g)")
    ax[0].set_xlim(0, 1.2 * logs.WQFM.max())
    ax[1].set_xlabel("CARB (g/g)")
    ax[1].set_xlim(0, 1.2 * logs.WCARB.max())
    ax[2].set_xlabel("CLAY (g/g)")
    ax[2].set_xlim(0, 1.2 * logs.WCLAY.max())
    ax[3].set_xlabel("PYR (g/g)")
    ax[3].set_xlim(0, 0.2)
    ax[4].set_xlabel("Mineralogy")
    ax[4].set_xlim(0, 1)
    ax[0].set_ylabel('Depth (m)');
    ax[1].set_yticklabels([]);
    ax[2].set_yticklabels([]);
    ax[3].set_yticklabels([]);
    ax[4].set_yticklabels([]);
    f.suptitle('Well: %s' % str(logs.iloc[0]['ID']), fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(fn_save, dpi=300)


def scatter_plot(savepath, fn, d_slice_all):
    plt.rcParams.update({'font.size': 17})
    RMSE_1 = np.sqrt(np.mean(np.square((d_slice_all['WQFM'] - d_slice_all['WQFM_pre']))));
    RR_1 = np.corrcoef(d_slice_all['WQFM'], d_slice_all['WQFM_pre'])[0, 1]
    RMSE_2 = np.sqrt(np.mean(np.square((d_slice_all['WCARB'] - d_slice_all['WCARB_pre']))));
    RR_2 = np.corrcoef(d_slice_all['WCARB'], d_slice_all['WCARB_pre'])[0, 1]
    RMSE_3 = np.sqrt(np.mean(np.square((d_slice_all['WCLAY'] - d_slice_all['WCLAY_pre']))));
    RR_3 = np.corrcoef(d_slice_all['WCLAY'], d_slice_all['WCLAY_pre'])[0, 1]
    RMSE_4 = np.sqrt(np.mean(np.square((d_slice_all['WPYR'] - d_slice_all['WPYR_pre']))));
    RR_4 = np.corrcoef(d_slice_all['WPYR'], d_slice_all['WPYR_pre'])[0, 1]

    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 13))
    ax1 = axs1[0, 0]
    ax1.scatter(d_slice_all['WQFM'], d_slice_all['WQFM_pre'], s=9, marker='^', c='g', )
    ax1.set_xlim(0, d_slice_all['WQFM'].max())
    ax1.set_ylim(0, d_slice_all['WQFM'].max())
    ax1.legend([f'RMSE: {RMSE_1:.4f}\n$R^{2}$: {RR_1:.4f}'], loc='upper left', title='QFM', shadow=True, facecolor='w',
               framealpha=1, frameon=True, handlelength=0, markerscale=0)
    ax1.set_xlabel(r'Real $(g/g)$')
    ax1.set_ylabel(r'Predicted $(g/g)$')
    ax1.grid(linestyle='--')
    ax2 = axs1[0, 1]
    ax2.scatter(d_slice_all['WCARB'], d_slice_all['WCARB_pre'], s=9, marker='o', c='r', )
    ax2.set_xlim(0, d_slice_all['WCARB'].max())
    ax2.set_ylim(0, d_slice_all['WCARB'].max())
    ax2.legend([f'RMSE: {RMSE_2:.4f}\n$R^{2}$: {RR_2:.4f}'], loc='upper left', title='CARB', shadow=True, facecolor='w',
               framealpha=1, frameon=True, handlelength=0, markerscale=0)
    ax2.set_xlabel(r'Real $(g/g)$')
    ax2.set_ylabel(r'Predicted $(g/g)$')
    ax2.grid(linestyle='--')
    ax3 = axs1[1, 0]
    ax3.scatter(d_slice_all['WCLAY'], d_slice_all['WCLAY_pre'], s=10, marker='.', c='b', )
    ax3.set_xlim(0, d_slice_all['WCLAY'].max())
    ax3.set_ylim(0, d_slice_all['WCLAY'].max())
    ax3.legend([f'RMSE: {RMSE_3:.4f}\n$R^{2}$: {RR_3:.4f}'], loc='upper left', title='CLAY', shadow=True, facecolor='w',
               framealpha=1, frameon=True, handlelength=0, markerscale=0)
    ax3.set_xlabel(r'Real $(g/g)$')
    ax3.set_ylabel(r'Predicted $(g/g)$')
    ax3.grid(linestyle='--')
    ax4 = axs1[1, 1]
    ax4.scatter(d_slice_all['WPYR'], d_slice_all['WPYR_pre'], s=9, marker='*', c='m', )
    ax4.set_xlim(0, d_slice_all['WPYR'].max())
    ax4.set_ylim(0, d_slice_all['WPYR'].max())
    ax4.legend([f'RMSE: {RMSE_4:.4f}\n$R^{2}$: {RR_4:.4f}'], loc='upper left', title='PYR', shadow=True, facecolor='w',
               framealpha=1, frameon=True, handlelength=0, markerscale=0)
    ax4.set_xlabel(r'Real $(g/g)$')
    ax4.set_ylabel(r'Predicted $(g/g)$')
    ax4.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f'scatter_{fn[0:-4]}.png'), dpi=300)


def DataAugment_Jon(X0):
    m, n = X0.shape
    X = np.empty((m, 3 * n))
    print('check df: ', type(X0), X0.shape, X0.head())
    for idx, row in X0.iterrows():
        print(idx, row)
    return X


def Las_slice(las):
    temp = las.df()
    try:
        temp1 = las.header['Tops']
        temp1 = las.header['T']
    except:
        temp1 = 'Error   888'

    temp2 = temp1.split('\n')
    tops = {re.split('   +', str0)[0]: float(re.split('   +', str0)[1]) for str0 in temp2}
    try:
        temp4 = temp[temp.index >= tops['MUSKWA']]
    except:
        temp4 = temp
    try:
        D_slice0 = temp4[temp4.index <= tops['KEG RIVER']]
    except:
        D_slice0 = temp4
    return D_slice0, tops


def idenfity_logs(d_slice0, GR_index, RD_index, DTP_index, DEN_index, PE_index):
    temp_str = ['DEPT']
    for idx, str in enumerate(GR_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 2: break
    for idx, str in enumerate(RD_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 3: break
    for idx, str in enumerate(DTP_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 4: break
    for idx, str in enumerate(DEN_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 5: break
    for idx, str in enumerate(PE_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 6: break
    temp_str.append('WQFM')
    for idx, str in enumerate(['WCARB', 'WCAR']):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 8: break
    for idx, str in enumerate(['WCLAY', 'WCLA']):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 9: break
    temp_str.append('WPYR')
    return temp_str


def idenfity_logs_extra(d_slice0, GR_index, RD_index, DTP_index, DEN_index):
    temp_str = ['DEPT']
    for idx, str in enumerate(GR_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 2: break
    for idx, str in enumerate(RD_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 3: break
    for idx, str in enumerate(DTP_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 4: break
    for idx, str in enumerate(DEN_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 5: break
    temp_str.append('WQFM')
    for idx, str in enumerate(['WCARB', 'WCAR']):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 8: break
    for idx, str in enumerate(['WCLAY', 'WCLA']):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 9: break
    temp_str.append('WPYR')
    return temp_str


def idenfity_logs_input(d_slice0, GR_index, RD_index, DTP_index, DEN_index, PE_index):
    temp_str = ['DEPT']
    for idx, str in enumerate(GR_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 2: break
    for idx, str in enumerate(RD_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 3: break
    for idx, str in enumerate(DTP_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 4: break
    for idx, str in enumerate(DEN_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 5: break
    for idx, str in enumerate(PE_index):
        if str in d_slice0.columns:
            temp_str.append(str)
        if len(temp_str) == 6: break
    return temp_str


def plot_corr(savepath, input_str, output_str, d_slice_all, X):
    plt.rcParams.update({'font.size': 15})
    colors_list = ['#78C850', '#F08030', '#6890F0', '#A8B820', '#F8D030', '#E0C068', '#C03028', '#F85888', '#98D8D8']
    fig = plt.figure(figsize=(19, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 5)
    ax = fig.add_subplot(gs[0, 0])
    sns.violinplot(y=input_str[0], data=d_slice_all, color=colors_list[0], linewidth=0.8, dodge=False)
    ax.set_xlabel("GR");
    ax.set_ylabel("(API)");
    ax = fig.add_subplot(gs[0, 1])
    sns.violinplot(y=input_str[1], data=d_slice_all[d_slice_all[input_str[1]] < 300][d_slice_all[input_str[1]] >= 0],
                   color=colors_list[1], linewidth=0.8, dodge=False)
    ax.set_xlabel("RD");
    ax.set_ylabel("(ohm.m)");
    ax = fig.add_subplot(gs[0, 2])
    sns.violinplot(y=input_str[2], data=d_slice_all, color=colors_list[2], linewidth=0.8, dodge=False)
    ax.set_xlabel("DTP");
    ax.set_ylabel("(usec/m)");
    ax = fig.add_subplot(gs[0, 3])
    sns.violinplot(y=input_str[3],
                   data=d_slice_all[d_slice_all[input_str[3]] < 3100][d_slice_all[input_str[3]] >= 2200],
                   color=colors_list[4], linewidth=0.8, dodge=False)
    ax.set_xlabel("DEN");
    ax.set_ylabel("(g/cc)");
    ax = fig.add_subplot(gs[0, 4])
    sns.violinplot(y=input_str[4], data=d_slice_all[d_slice_all[input_str[4]] < 8][d_slice_all[input_str[4]] >= 2],
                   color=colors_list[6], linewidth=0.8, dodge=False)
    ax.set_xlabel("PE");
    ax.set_ylabel("(barns/e)");
    plt.savefig(os.path.join(savepath, 'violin_input.png'), dpi=300)

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 4)
    ax = fig.add_subplot(gs[0, 0])
    sns.violinplot(y=output_str[0], data=d_slice_all, color=colors_list[0], linewidth=0.8, dodge=False)
    ax.set_xlabel("QFM");
    ax.set_ylabel("(g/g)");
    ax = fig.add_subplot(gs[0, 1])
    sns.violinplot(y=output_str[1], data=d_slice_all[d_slice_all[output_str[1]] < 1][d_slice_all[output_str[1]] >= 0],
                   color=colors_list[1], linewidth=0.8, dodge=False)
    ax.set_xlabel("CARB");
    ax.set_ylabel("(g/g)");
    ax = fig.add_subplot(gs[0, 2])
    sns.violinplot(y=output_str[2], data=d_slice_all, color=colors_list[2], linewidth=0.8, dodge=False)
    ax.set_xlabel("CLAY");
    ax.set_ylabel("(g/g)");
    ax = fig.add_subplot(gs[0, 3])
    sns.violinplot(y=output_str[3], data=d_slice_all[d_slice_all[output_str[3]] < 0.2][d_slice_all[output_str[3]] >= 0],
                   color=colors_list[4], linewidth=0.8, dodge=False)
    ax.set_xlabel("PYR");
    ax.set_ylabel("(g/g)");
    # ax.set_ylim(2,8)
    plt.savefig(os.path.join(savepath, 'violin_output.png'), dpi=300)
    #plt.show();    exit()

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(11, 5))
    ax = sns.distplot(X[:, 0], hist=False, kde=True, kde_kws=dict(linewidth=0.6), color='r');
    ax = sns.distplot(X[:, 1], hist=False, kde=True, kde_kws=dict(linewidth=0.6), color='b');
    ax = sns.distplot(X[:, 2], hist=False, kde=True, kde_kws=dict(linewidth=0.6), color='g');
    ax = sns.distplot(X[:, 3], hist=False, kde=True, kde_kws=dict(linewidth=0.6), color='c');
    ax = sns.distplot(X[:, 4], hist=False, kde=True, kde_kws=dict(linewidth=0.6), color='m');
    x1 = ax.lines[0].get_xydata()[:, 0];
    y1 = ax.lines[0].get_xydata()[:, 1];
    x2 = ax.lines[1].get_xydata()[:, 0];
    y2 = ax.lines[1].get_xydata()[:, 1];
    x3 = ax.lines[2].get_xydata()[:, 0];
    y3 = ax.lines[2].get_xydata()[:, 1];
    x4 = ax.lines[3].get_xydata()[:, 0];
    y4 = ax.lines[3].get_xydata()[:, 1];
    x5 = ax.lines[4].get_xydata()[:, 0];
    y5 = ax.lines[4].get_xydata()[:, 1];
    ax.fill_between(x1, y1, color='r', alpha=0.3);
    ax.fill_between(x2, y2, color='b', alpha=0.3);
    ax.fill_between(x3, y3, color='g', alpha=0.3);
    ax.fill_between(x4, y4, color='c', alpha=0.3);
    ax.fill_between(x5, y5, color='m', alpha=0.3);
    plt.xlim(-3, 3);
    plt.ylim(0, 1.75);
    plt.xlabel('Standardized value');
    plt.ylabel('Density')
    plt.legend(input_str, fancybox=True, shadow=False, facecolor='w', framealpha=1, frameon=True);
    plt.grid(linestyle='--')
    plt.savefig(os.path.join(savepath, 'dist_input.png'), dpi=300)

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(7, 5))
    mask = np.triu(np.ones_like(d_slice_all[input_str].corr(), dtype=np.bool))
    ax1 = sns.heatmap(d_slice_all[input_str].corr(), mask=mask, annot=True, cmap='BrBG');
    plt.savefig(os.path.join(savepath, 'corr_input.png'), dpi=300)

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(7, 5))
    mask = np.triu(np.ones_like(d_slice_all[output_str].corr(), dtype=np.bool))
    ax1 = sns.heatmap(d_slice_all[output_str].corr(), mask=mask, annot=True, cmap='BrBG');
    plt.savefig(os.path.join(savepath, 'corr_output.png'), dpi=300)
    # plt.show();
