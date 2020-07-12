# FashionIQChallenge2020
__2nd Place Team's Solution for Fashion IQ Challenge 2020__
+ CodaLab: https://competitions.codalab.org/competitions/22946#results
+ Workshop page: https://sites.google.com/view/cvcreative2020/fashion-iq

<img src="https://github.com/nashory/FashionIQChallenge2020/blob/master/img/main_diagram.png?raw=true"></img>



## (1) Environment Setup
+ Prepare Data:    
Download Dataset from [here](https://github.com/XiaoxiaoGuo/fashion-iq)   

Image download script has been added. (12/07/2020)
~~~ bash
run_download_image.sh
~~~ 

The data structure should look like:
~~~bash
cd ./ours/train
dataset
   ├──image_splits
   │       ├──split.dress.train.json
   │       ├──...
   │       ├──split.toptee.test.json
   │
   ├──captions
   │       ├──cap.dress.train.json
   │       ├──...
   │       ├──cap.toptee.test.json
   │
   ├──images
           ├──B000OHM9FI.jpg
           ├──...
           ├──B000M89C40.jpg

~~~

+ Install required packages using virtualenv
~~~
python3 -m virtualenv --python=python3 py3
. py3/bin/activate
pip install -r requirements.txt
~~~

+ Download required files: word2vec, sentence_embedding, image_emedding   
Download from here: [google drive](https://drive.google.com/drive/folders/1wYpxqzPLw0r383Gfxysp7d1UEEY6mzne?usp=sharing)    
~~~
cd ./ours/train
tar -xvf assets.tar
~~~


## (2) Train a Single Model (TIRG, Best Score: 37.18)

Run the script below.
<details>   
    <summary>overall score curve</summary>   
    <img src="https://github.com/nashory/FashionIQChallenge2020/blob/master/img/overall_score.png?raw=true"></img>
</details>   

<details>   
    <summary>loss curve</summary>   
    <img src="https://github.com/nashory/FashionIQChallenge2020/blob/master/img/loss.png?raw=true"></img>
</details>   

~~~
cd ours/train
python3 main.py \
    --warmup \
    --gpu_id '0' \
    --method 'tirg' \
    --text_method 'lstm-gru' \
    --expr_name 'devel' \
    --data_root './dataset' \
    --backbone 'resnet152' \
    --fdims 2048 \
    --epochs 100 \
    --batch_size 32 \
    --image_size 224 \
    --normalize_scale 5.0 \
    --lr 0.00011148 \
    --lrp 0.48 \
    --lr_decay_steps "10,20,30,40,50,60,70" \
    --lr_decay_factor 0.4747
~~~


## (3) Ensemble Scores using Bayesian Optimization
__(i) get the score output on test/val:__  
~~~
cd ours/tools
python3 get_score.py --data_root './dataset' --expr_name 'devel'
~~~

__(ii) get emsemble score using Bayesian Optimization provided by hyperopt:__  
Check the name of output score saved in `ours/tools/output_score`.   
if the candidate models are: ['20200605122245_devel', '20200605152325_devel']
~~~
python3 optimize_score.py --data_root './dataset' --repos '20200605122245_devel,20200605152325_devel'
~~~

## Authors
+ [Minchul Shin, Search Solutions](https://github.com/nashory)
+ [Yoonjae Cho, NAVER/LINE Vision](https://github.com/yoonjaecho)
+ [Seongwuk Hong, NAVER/LINE Vision](https://github.com/wookie0)
