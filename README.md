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

## Reference
[amp_gan](https://github.com/lsbnb/amp_gan)
#### Modifications
* support training on CUDA
* 5-dimension AAF encoding
* RMSProp optimizer

## To cite
```
@article{Dong2024.02.23.581845,
  author = {Ruihan Dong and Rongrong Liu and Ziyu Liu and Yangang Liu and Gaomei Zhao and Honglei Li and Shiyuan Hou and Xiaohan Ma and Huarui Kang and Jing Liu and Fei Guo and Ping Zhao and Junping Wang and Cheng Wang and Xingan Wu and Sheng Ye and Cheng Zhu},
  title = {Exploring the repository of de novo designed bifunctional antimicrobial peptides through deep learning},
  elocation-id = {2024.02.23.581845},
  year = {2024},
  doi = {10.1101/2024.02.23.581845},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2024/02/24/2024.02.23.581845},
  journal = {bioRxiv}
}
```
