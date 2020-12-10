import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import ortho_group
import datetime
import sys
import csv
import os
import argparse
from tensorflow.keras.datasets import mnist

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--n_features', default=784, type=int)
parser.add_argument('--m_samples', default=60000, type=int)
parser.add_argument('--p_components', default=100, type=int)
parser.add_argument('--save_interval', default=100, type=int)
parser.add_argument('--epochs', default=200000, type=int)
parser.add_argument('--threshold', default=0.01, type=float)
parser.add_argument('--lr', default=[1e-3, 1e-4, 1e-5, 1e-6], type=list)
parser.add_argument('--lr_sche', default=[1e-2, 1e-3, 1e-4], type=list)
parser.add_argument('--stddev', default=1e-7, type=float)
parser.add_argument('--loss', default="new", choices=["new", "old"], type=str)
args = parser.parse_args()

# PRG seed
seed = args.seed

# Numpy general config
np.set_printoptions(precision=15)
np.random.seed(seed)

# Tensorflow general config
tf.executing_eagerly()
tf.random.set_seed(seed)
tfpd = tfp.distributions

# General Parameters
n = args.n_features  # Dimension of input and output data
p = args.p_components  # Dimension of encoder output (p<n)
m = args.m_samples  # Dimension of samples
save_interval = args.save_interval
EPOCHS = args.epochs

Tp = tf.constant(np.diagflat(np.fromfunction(lambda i, j: (p - i), (p, 1))), dtype=tf.float64)
Sp = tf.constant(np.fromfunction(lambda i, j: p - np.maximum(i, j), (p, p)), dtype=tf.float64)
Tp_diag = tf.linalg.diag_part(Tp)
Ip = tf.eye(p, dtype=tf.float64)

result_dir = "/results/mnist/" + "/" + str(n) + "_" + str(m) + "_" + str(p) + args.loss + "/"

# directory for saving results
com_dir = "/results/mnist" + "/" + str(n) + "_" + str(m) + "_" + str(p) + args.loss + "/"
# directory for saving component
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(com_dir):
    os.makedirs(com_dir)


def cal_re_error(X,Com):
    n1 = X.shape[0]
    n2 = X.shape[1]

    temp1 = Com @ tf.transpose(Com)
    temp2 = tf.transpose(temp1 @ tf.transpose(X))
    temp3 = tf.linalg.norm(X-temp2)
    re_error = temp3*temp3
    re_error /= int(n1)
    re_error /= int(n2)
    return re_error


class Data(object):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')
    x_train = x_train.astype('float64')
    x_train /= 255
    X = x_train.reshape(60000, 784)
    mean_data = np.mean(X, axis=0)
    X = X - mean_data
    X = tf.transpose(X)
    Y = X

    # Evaluate covariances
    SigmaYX = Y @ tf.transpose(X)
    SigmaXY = tf.transpose(SigmaYX)
    SigmaXX = X @ tf.transpose(X)

    SigmaYY_norm2 = tf.linalg.trace(Y @ tf.transpose(Y))
    Sigma = SigmaXY


class Loss(object):
    @staticmethod
    def LossFunc2(A, B):  # Our loss function
        loss = tf.linalg.trace(tf.transpose(Data.Y) @ (p * Data.Y - 2 * (Tp_diag * A) @ B @ Data.X)) \
               + tf.linalg.trace((Sp * (tf.transpose(A) @ A)) @ B @ Data.SigmaXX @ tf.transpose(B))
        loss = 1 / m * loss
        return loss

    @staticmethod
    def LossFunc3(A, B):  # Simplified version of LossFunc2
        loss = p * Data.SigmaYY_norm2
        loss = loss - 2 * tf.linalg.trace((Tp_diag * A) @ B @ Data.SigmaXY) \
               + tf.linalg.trace((Sp * (tf.transpose(A) @ A)) @ B @ Data.SigmaXX @ tf.transpose(B))
        loss = 1 / m * loss
        return loss

    @staticmethod
    def LossFunc4(A, B):  # The good old L2 loss
        loss = tf.Variable(np.zeros((1, 1)), dtype='float64')
        loss = 1 / m * tf.norm(Data.Y - A @ B @ Data.X) ** 2
        return loss

    @staticmethod
    def EvalLossFunc(A, B):
        if args.loss == "new":
            return Loss.LossFunc3(A, B)
        elif args.loss == "old":
            return Loss.LossFunc4(A, B)


class DesSol(object):  # Handles the analytical solution based on Eigenvalue decomposition
    U = tf.linalg.eigh(Data.Sigma)
    A = U[1][0:n, n - 1:n - p - 1:-1]
    B = tf.transpose(A)
    DesLoss = Loss.EvalLossFunc(A, B)

    @staticmethod
    def GetAltDesSol(C):  # e.x. tf.constant([[3.0, 0], [0, 5.0]], dtype=tf.float64)
        DesA2 = DesSol.A @ C
        DesB2 = tf.matrix_inverse(C) @ DesSol.B
        return Loss.EvalLossFunc(DesA2, DesB2)


class Model(object):
    def __init__(self):
        te = tf.random.normal([n, p], mean=0.0, stddev=args.stddev, dtype=tf.float64)
        self.A = tf.Variable(te, name='A', trainable=True)
        self.B = tf.Variable(tf.transpose(te), name='B', trainable=True)

    def EvalLoss(self):
        return Loss.EvalLossFunc(self.A, self.B)

    @staticmethod
    def AnalyticalGradient(A, B):
        dB = -2 / m * (Tp @ tf.transpose(A) @ Data.SigmaYX - (Sp * (tf.transpose(A) @ A)) @ B @ Data.SigmaXX)
        dA = -2 / m * (Data.SigmaYX @ tf.transpose(B) @ Tp - A @ (
                Sp * (B @ Data.SigmaXX @ tf.transpose(B))))
        grad = ((dA, A), (dB, B))
        return grad


# Create the model
mdl = Model()

# Training the model
epochs = range(EPOCHS)

lr = tf.Variable(args.lr[0])
optimizer = tf.optimizers.Adam(learning_rate=lr)
heads = ['epoch', 'loss', 'lossD', "grad_norm", "diagmin", "offdiagmax", "normalized_re_error", "count_TP", "count_FP",
         "ratio_TP", "ratio_FP", "ratio_Total", "time_used"]
with open(result_dir + "method1.csv", "w") as output:
    writer = csv.writer(output, lineterminator=",")
    for val in heads:
        writer.writerow([val])
with open(result_dir + "method1.csv", "a") as output:
    output.write("\n")

start_time = datetime.datetime.now()
end_time = datetime.datetime.now()
total_time = end_time - start_time
start_time = datetime.datetime.now()

for epoch in epochs:

    if args.loss == "old":
        L = optimizer.compute_gradients(loss=mdl.EvalLoss, var_list=[mdl.A, mdl.B])
    elif args.loss == "new":
        L = mdl.AnalyticalGradient(mdl.A, mdl.B)
    optimizer.apply_gradients(L)
    if epoch % save_interval == 0:

        lo = mdl.EvalLoss()
        for j, sche in enumerate(args.lr_sche):
            if (lo - DesSol.DesLoss) / args.p_components < sche:
                lr.assign(args.lr[j + 1])

        end_time_temp = datetime.datetime.now()

        CM = tf.abs(tf.transpose(DesSol.A) @ mdl.A / tf.norm(mdl.A, axis=0, keepdims=True))
        diagmin = tf.reduce_min(tf.linalg.diag_part(CM))
        offdiagmax = tf.reduce_max(tf.linalg.set_diag(CM, tf.constant(np.zeros(p), dtype=tf.float64)))
        print('Epoch %2d:, loss=%2.15f' % (epoch, lo))
        lossD_temp = (lo - DesSol.DesLoss)
        print('LossD=%2.15f' % lossD_temp)
        grad_norm_temp = tf.norm(L[0][0])
        print('grad_norm=%2.15f' % grad_norm_temp)
        print('diagmin=%2.15f,   offdiagmax=%2.15f' % (diagmin, offdiagmax))
        method1_com_temp = tf.math.l2_normalize(mdl.A, axis=0)
        re_error_temp = cal_re_error(tf.transpose(Data.X), method1_com_temp)
        print("re_error_temp", re_error_temp)
        NumOfTrueDiag_temp = np.sum(tf.linalg.diag_part(CM) > 1 - args.threshold)
        print('NumOfTrueDiag=%4d' % NumOfTrueDiag_temp)
        NumOfFalseOffDiag_temp = np.sum(
            tf.linalg.set_diag(CM, tf.constant(np.zeros(p), dtype=tf.float64)) > 1 - args.threshold)
        print('NumOfFalseOffDiag=%4d' % NumOfFalseOffDiag_temp)

        ratio_TP = NumOfTrueDiag_temp / args.p_components
        ratio_FP = NumOfFalseOffDiag_temp / args.p_components
        time_temp = end_time_temp - start_time
        total_time += time_temp

        result_temp = [epoch, round(lo.numpy(), 8),
                       round(lossD_temp.numpy(), 8), round(grad_norm_temp.numpy(), 8), round(diagmin.numpy(), 8),
                       round(offdiagmax.numpy(), 8),
                       round(re_error_temp.numpy(), 8), NumOfTrueDiag_temp, NumOfFalseOffDiag_temp, round(ratio_TP, 8),
                       round(ratio_FP, 8), round(ratio_TP + ratio_FP, 8),
                       total_time]

        with open(result_dir + "method1.csv", "a") as output:
            writer = csv.writer(output, lineterminator=',')
            for val in result_temp:
                writer.writerow([val])
        with open(result_dir + "method1.csv", "a") as output:
            output.write("\n")

        start_time = datetime.datetime.now()
        if NumOfTrueDiag_temp == args.p_components:
            break
        np.savetxt(com_dir + "method_component.txt", tf.math.l2_normalize(mdl.A, axis=0).numpy())
