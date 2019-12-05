# shallow-fake
## A fake character generator based on wgan-gp 
### Dataset
Dataset can be generated in ./make_dataset/make_dataset.py. The code and chinese text file is based on https://github.com/marcin7Cd/variant-of-CPPN-GAN/blob/master/chinese.py. To creat dataset from fonts, run: 
```bash
python3 make_dataset.py [target_forder_name]
```  
A target forder should have [target]_fonts to store fonts and [target_text] to store text for sampling. Chinese fonts can be downloaded from  https://www.kaggle.com/dylanli/chinesecharacter. Japanese fonts can be downloaded from https://www.freejapanesefont.com/.
