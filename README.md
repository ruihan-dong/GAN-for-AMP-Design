# GAN for AMP Design
A GAN-based generator for de novo AMP design



![image](https://github.com/ruihan-dong/GAN-for-AMP-Design/blob/main/GAN-Framework.png)

## Usage
### Train
```bash
chmod +x run.sh
./run.sh
```
* remember to check the paths within the script.

### Generate
```bash
python generate.py -b batch_size
```
* batch_size: the number of sequences to generate
* The generator weight files are in `./model`, `AMP_generator.pkl` is the generator v1 and `AMP_AVP_generator.pkl` is the generator v2.

## Reference
[amp_gan](https://github.com/lsbnb/amp_gan)
#### Modifications
* support training on CUDA
* 5-dimension AAF encoding
* RMSProp optimizer

## To cite
```
@article{Dong_elife_2024, 
    title = {Exploring the repository of de novo designed bifunctional antimicrobial peptides through deep learning}, 
    url = {http://dx.doi.org/10.7554/eLife.97330.1}, 
    DOI = {10.7554/elife.97330.1}, 
    publisher = {eLife Sciences Publications, Ltd}, 
    author = {Dong, Ruihan and Liu, Rongrong and Liu, Ziyu and Liu, Yangang and Zhao, Gaomei and Li, Honglei and Hou, Shiyuan and Ma, Xiaohan and Kang, Huarui and Liu, Jing and Guo, Fei and Zhao, Ping and Wang, Junping and Wang, Cheng and Wu, Xingan and Ye, Sheng and Zhu, Cheng}, 
    year = {2024}, 
    volume = {13},
    pages = {RP97330},
    month = may, 
    journal = {eLife}
}
```
