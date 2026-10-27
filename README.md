# ComfyUI_Sonic
[Sonic](https://github.com/jixiaozhong/Sonic) is a method about ' Shifting Focus to Global Audio Perception in Portrait Animation',you can use it in comfyUI

# Update
* 去掉diffuser和transformer的版本依赖，改成comfyUI的patcher封装/change diffuser and transformer version dependency to patcher in comfyUI


# 1. Installation

In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Sonic.git
```

# 2. Requirements  
```
pip install -r requirements.txt
```

# 3.Models
* 3.1 download  checkpoints  from [google](https://drive.google.com/drive/folders/1oe8VTPUy0-MHHW2a_NJ1F8xL-0VN5G7W) 从Google下载必须的模型,文件结构如下图
* 3.2 download [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny/tree/main)
* 3.3 SVD checkpoints  [svd_xt.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  or [svd_xt_1_1.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)    
```
--  ComfyUI/models/sonic/
    |-- audio2bucket.pth
    |-- audio2token.pth
    |-- unet.pth
    |-- yoloface_v5m.pt
    |-- whisper-tiny/
        |--config.json
        |--model.safetensors
        |--preprocessor_config.json
    |-- RIFE/
        |--flownet.pkl
--   ComfyUI/models/checkpoints
    ├── svd_xt.safetensors  or  svd_xt_1_1.safetensors
```

# 4 Example
* new
![](https://github.com/smthemex/ComfyUI_Sonic/blob/main/example_workflows/example.png)


# 5 Citation
```
@article{ji2024sonic,
  title={Sonic: Shifting Focus to Global Audio Perception in Portrait Animation},
  author={Ji, Xiaozhong and Hu, Xiaobin and Xu, Zhihong and Zhu, Junwei and Lin, Chuming and He, Qingdong and Zhang, Jiangning and Luo, Donghao and Chen, Yi and Lin, Qin and others},
  journal={arXiv preprint arXiv:2411.16331},
  year={2024}
}

@article{ji2024realtalk,
  title={Realtalk: Real-time and realistic audio-driven face generation with 3d facial prior-guided identity alignment network},
  author={Ji, Xiaozhong and Lin, Chuming and Ding, Zhonggan and Tai, Ying and Zhu, Junwei and Hu, Xiaobin and Luo, Donghao and Ge, Yanhao and Wang, Chengjie},
  journal={arXiv preprint arXiv:2406.18284},
  year={2024}
}
```
