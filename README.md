# CDAPM

## Environment Requirement
The code has been tested running under Python 2.7. The required packages are as follows:
- tensorflow-gpu == 1.14.0
- keras == 2.2.4
- keras_self_attention == 0.49.0 <br/>https://pypi.org/project/keras-self-attention/
- keras_pos_embd == 0.8.0 <br/>https://pypi.org/project/keras-pos-embd/0.8.0/

## Project Structure

- **/model**
    - **CDAPM.py** *#The implementation of CDAPM*
    - **EmbeddingMatrix.py** *#The Embedding for text-embedding for a 3-d vector to a 2-d vector*
    - **mode_normalization.py** *# The implementation of Mode Normalization * 
- **/data** *# Path of New York Foursquare and Los Angelos Geo-Tweets datasets*
    - **la_tweets1100000_cikm.txt** *# Los Angelos Geo-Tweets*
    - **tweets-cikm.txt** *# New York Foursquare check-ins*
    - **venues.txt** *# New York POIs*
- **/word_vec**
    - **glove.twitter.27B.50d.txt** *# Pretrained Glove word embeddings*
- **/features** *# The processed trajectory samples and features by geo_data_decoder.py*
    - **features&index_seg_gride_fs** *# New York*
    - **features&index_seg_gride_la** *# Los Angelos*

- **config.py** *# It contains all the configurations in CDAPM*
- **eval_tools.py** *# It contains helper functions and evaluation tools*
- **geo_data_decoder.py** *# Preprocessing for both New York Foursquare and Los Angelos Geo-Tweets*
- **read_features.py** *# Get the features and samples that model need without running geo_data_decoder.py*
- **train.py** *# Training procedure of CDAPM. This is the entrace of CDAPM model.*
## Usage
train and evaluate model :
> ```python
> python train.py 
> ```

## Dataset and external data
Some large files are not available in this repository.Please download on link: 链接: https://pan.baidu.com/s/1YsonkFBk2pN-PUJOC-Bbyg  password: ki82
- **/data** *# Path of New York Foursquare and Los Angelos Geo-Tweets datasets*
    - **la_tweets1100000_cikm.txt** *# Los Angelos Geo-Tweets*
    - **tweets-cikm.txt** *# New York Foursquare check-ins*
    - **venues.txt** *# New York POIs*
- **/word_vec**
    - **glove.twitter.27B.50d.txt** *# Pretrained Glove word embeddings*

