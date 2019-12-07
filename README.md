# shallow-fake
## A fake character generator based on wgan-gp 
### Dataset
Dataset can be generated in ./make_dataset/make_dataset.py. The code and chinese text file is derived from https://github.com/marcin7Cd/variant-of-CPPN-GAN/blob/master/chinese.py. To creat dataset from fonts, run: 
```bash
python3 make_dataset.py [target_forder_name]
```  
A target forder should have [target]_fonts to store fonts and [target_text] to store text for sampling. Chinese fonts can be downloaded from  https://www.kaggle.com/dylanli/chinesecharacter. Japanese fonts can be downloaded from https://www.freejapanesefont.com/.

### Training
The code for training the model derived from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2. The model we use is wgan-gp. 

### Evaluation
We used Frechet Inception Distance to evaluate our model. The code for creating npy derived from https://blog.csdn.net/yql_617540298/article/details/82747789. The code for calculateing fid derived from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/.

### Usage
