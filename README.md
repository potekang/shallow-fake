# shallow-fake
![Sample output](https://github.com/potekang/shallow-fake/blob/master/output_sample.jpg)
## A fake character generator based on wgan-gp 
### Dataset
Dataset can be generated in ./make_dataset/make_dataset.py. The code and chinese text file is derived from https://github.com/marcin7Cd/variant-of-CPPN-GAN/blob/master/chinese.py. 

### Training
The code for training the model derived from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2. The model we used is wgan-gp. 

### Evaluation
We used Frechet Inception Distance to evaluate our model. The code for creating npy derived from https://blog.csdn.net/yql_617540298/article/details/82747789. The code for calculateing fid derived from https://github.com/bioinf-jku/TTUR.

### Usage
#### Create dataset from fonts: 
```bash
python3 make_dataset.py [target_forder_name]
```  
A target forder should have [target]_fonts to store fonts and [target_text] to store text for sampling. Chinese fonts can be downloaded from  https://www.kaggle.com/dylanli/chinesecharacter. Japanese fonts can be downloaded from https://www.freejapanesefont.com/. Images should moved to /shallow-fake/dataset/ directory.

#### Sample 1000 images for fid evaluation:
```bash
python3 gen_eval_img.py
```  
Dataset specification can be modified in the python file.
#### Training:
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset=[language] --epoch=[epoch_num] --n_d=5
```  
Our setting is:
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset=japanese --epoch=800 --n_d=5
```
The fid score after certain epoch will be saved in ./fid_status.txt.

The model with lowest fid score will be saved in related folder in ./output
