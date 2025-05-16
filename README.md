# ComfyUI_Sonic
[Sonic](https://github.com/jixiaozhong/Sonic) is a method about ' Shifting Focus to Global Audio Perception in Portrait Animation',you can use it in comfyUI

# Update
* some guys cuda must use  cuda:0,so fix it. 修复有些人的电脑必须用cuda:0，否则会报错的错误。
* fix bf16 error,fix 12GVRAM maybe OOM when first run,fix MPS device error,修复bf16无法使用的错误，修复12GVram首次加载时容易OOM的问题，修复MAC的MPS支持。

# 1. Installation

In the ./ComfyUI/custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Sonic.git
```
---

# 2. Requirements  

```
pip install -r requirements.txt
```

# 3.Model
* 3.1.1 download  checkpoints  from [google](https://drive.google.com/drive/folders/1oe8VTPUy0-MHHW2a_NJ1F8xL-0VN5G7W) 从Google下载必须的模型,文件结构如下图
* 3.1.2 download [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny/tree/main)
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
```
*  3.2 SVD checkpoints  [svd_xt.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  or [svd_xt_1_1.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)    

```
--   ComfyUI/models/checkpoints
    ├── svd_xt.safetensors  or  svd_xt_1_1.safetensors
```

# Example
* new
![](https://github.com/smthemex/ComfyUI_Sonic/blob/main/example_workflows/example_0516.png)

* old
![](https://github.com/smthemex/ComfyUI_Sonic/blob/main/example_workflows/exampleB.png)
* old
![](https://github.com/smthemex/ComfyUI_Sonic/blob/main/example_workflows/exampleA.png)

# Previous update
* Replace 'frame number' with 'duration',you can use it to change 'infer audio seconds'. 使用duration替换frame number选项，用于控制输出音频的长度(单位为秒），注意因为实际对比长度是音频振幅数组，不是百分比精准；
* Fixed the bug of batch mismatch when the frame rate is not 25.修复帧率不是25时，batch不匹配的bug。
* Change the model loading to a monolithic SVD model, 模型加载改为单体SVD模型；  
* Support output of non square images，OOM 支持非正方形图片的输出，容易OOM；
* image_size is used to control the minimum size of the output image. If OOM, please reduce this value ,image_size用于控制输出图片的最小尺寸，如果OOM请调小这个数值；
* 感谢@civen-cn 提交的PR


# Citation
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
