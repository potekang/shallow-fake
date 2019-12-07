
import functools

import imlib as im
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import numpy as np
import tqdm
import cv2
import from_png2npy
import fid_keras as fid
#import fid

import module

#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset=japanese --epoch=800 --n_d=5

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

# command line & parameters
py.arg('--dataset', default='chinese')
py.arg('--batch_size', type=int, default=64)
py.arg('--epochs', type=int, default=25)
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--n_d', type=int, default=1)  # # d updates per g update
py.arg('--z_dim', type=int, default=128)
py.arg('--gradient_penalty_weight', type=float, default=10.0)

args = py.args()

adversarial_loss_mode = 'wgan'
gradient_penalty_mode = 'wgan-gp'

# output_dir
experiment_name = '%s_%s' % (args.dataset, adversarial_loss_mode)
if gradient_penalty_mode != 'none':
    experiment_name += '_%s' % gradient_penalty_mode
output_dir = py.join('output', experiment_name)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)



# ==============================================================================
# =                               make dataset                                 =
# ==============================================================================

def make_custom_datset(img_paths, batch_size, resize=64, drop_remainder=True, shuffle=True, repeat=1):
    @tf.function
    def _map_fn(img):
        #img = ...  # custom preprocessings, should output img in [0.0, 255.0]
        img = tf.image.resize(img, [resize, resize])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = tl.disk_image_batch_dataset(img_paths,
                                          batch_size,
                                          drop_remainder=drop_remainder,
                                          map_fn=_map_fn,
                                          shuffle=shuffle,
                                          repeat=repeat)
    img_shape = (resize, resize, 1)
    len_dataset = len(img_paths) // batch_size

    return dataset, img_shape, len_dataset

# ==============================================================================
# =                                    data                                    =
# ==============================================================================

# setup dataset
dataset_path = './dataset/'+args.dataset
img_paths = py.glob(dataset_path, '*.png') # image paths of custom dataset
dataset, shape, len_dataset = make_custom_datset(img_paths, args.batch_size)
n_G_upsamplings = n_D_downsamplings = 4  # 3 for 32x32 and 4 for 64x64


# ==============================================================================
# =                                   model                                    =
# ==============================================================================

# setup the normalization function for discriminator
d_norm = 'layer_norm'

# networks
G = module.ConvGenerator(input_shape=(1, 1, args.z_dim), output_channels=shape[-1], n_upsamplings=n_G_upsamplings, name='G_%s' % args.dataset)
D = module.ConvDiscriminator(input_shape=shape, n_downsamplings=n_D_downsamplings, norm=d_norm, name='D_%s' % args.dataset)
G.summary()
D.summary()

# adversarial_loss_functions
d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(adversarial_loss_mode)

G_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=args.lr, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G():
    with tf.GradientTape() as t:
        z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
        x_fake = G(z, training=True)
        x_fake_d_logit = D(x_fake, training=True)
        G_loss = g_loss_fn(x_fake_d_logit)

    G_grad = t.gradient(G_loss, G.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G.trainable_variables))

    return {'g_loss': G_loss}


@tf.function
def train_D(x_real):
    with tf.GradientTape() as t:
        z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
        x_fake = G(z, training=True)

        x_real_d_logit = D(x_real, training=True)
        x_fake_d_logit = D(x_fake, training=True)

        x_real_d_loss, x_fake_d_loss = d_loss_fn(x_real_d_logit, x_fake_d_logit)
        gp = gan.gradient_penalty(functools.partial(D, training=True), x_real, x_fake, mode=gradient_penalty_mode)

        D_loss = (x_real_d_loss + x_fake_d_loss) + gp * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D.trainable_variables))

    return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}


@tf.function
def sample(z):
    return G(z, training=False)

def read(dir):
    text = 999
    try:
        with open(dir, "r") as f:
            text =  f.read().replace('\n', '') 
    except Exception as e:
        print(e)
    return text

def write(f, text):
    text = str(text)
    try:
        with open(f, "w") as f:
            f.write(text)
    except Exception as e:
        print(e)
# ==============================================================================
# =                                 evaluation                                 =
# ==============================================================================
def eval_fid():
    train_dir = "./evaluation/training"
    gen_dir= "./evaluation/generated"
    train_data = from_png2npy.getFileArr(train_dir,"train_data")
    generated = from_png2npy.getFileArr(gen_dir,"generated")   
    TRAIN_DIR = "./evaluation/train_data.npy"
    GEN_DIR = "./evaluation/generated.npy"
    fid_val = fid.cal_fid(TRAIN_DIR, GEN_DIR)
    return fid_val

# ==============================================================================
# =                                    run                                     =
# ==============================================================================
#optimal fid 
opt_fid = 999.0
opt_fid_dir = py.join(output_dir, 'opt_checkpoint/opt_fid.txt')
# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G=G,
                                D=D,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)
# best_checkpoint
min_fid = 9999;
opt_checkpoint = tl.Checkpoint(dict(G=G,
                                D=D,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'opt_checkpoint'),
                           max_to_keep=1)

try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

try:  # restore opt_checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
    opt_fid = float(read(opt_fid_dir))
    #print(opt_fid)
    #opt_fid = 500.1
    #write(opt_fid_dir, opt_fid)
    #opt_fid = read(opt_fid_dir)
    #print(opt_fid)
	
except Exception as e:
    print("restore opt checkopint failed")
    print(e)



# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main 


z = tf.random.normal((100, 1, 1, args.z_dim))  # a fixed noise for sampling
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for x_real in tqdm.tqdm(dataset, desc='Inner Epoch Loop', total=len_dataset):
            D_loss_dict = train_D(x_real)
            tl.summary(D_loss_dict, step=D_optimizer.iterations, name='D_losses')

            if D_optimizer.iterations.numpy() % args.n_d == 0:
                G_loss_dict = train_G()
                tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                x_fake = sample(z)
                img = im.immerge(x_fake, n_rows=10).squeeze()
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)
        #calculate fid after 500 epoch
        #be nice to me graphic card
        if ep > 700:
            #generate fake pics
            samples_dir = "./evaluation/generated"
            py.mkdir(samples_dir)
            for i in range(0,1000):
                z = tf.random.normal(shape=(1, 1, 1, args.z_dim))
                x_fake = G(z, training=False)
                img = im.immerge_(x_fake).squeeze()
                im.imwrite( img, py.join(samples_dir, 'fake%03d.jpg' %i))
            #generate npy & compare & savefid
            temp_fid = eval_fid()
            if(temp_fid < opt_fid):
                opt_fid = temp_fid
                opt_checkpoint.save(ep)
                write(opt_fid_dir, opt_fid)
                print('opt_fid updated: ', opt_fid)
 
# samples_dir = "./evaluation/generated"
# py.mkdir(samples_dir)
# for i in range(0,1000):
#     #z = tf.random.normal(shape=(args.batch_size, 1, 1, args.z_dim))
#     z = tf.random.normal(shape=(1, 1, 1, args.z_dim))
#     x_fake = G(z, training=False)
#     img = im.immerge_(x_fake).squeeze()
#     im.imwrite( img, py.join(samples_dir, 'fake%03d.jpg' %i))


