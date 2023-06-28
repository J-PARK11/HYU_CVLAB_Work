# personal_junhyeokpark
Project: Video Frame Interpolation, Semantic Segmentation

## 📑 Index
* Optical Flow + Monocular Depth Guide Video Frame Interpolation
* Dynamic Object Semantic Segmentation

## 🎞️ Optical Flow + Monocular Depth Guide Video Frame Interpolation

### Link
Notion: [Video Frame Interpolation Notion](https://www.notion.so/Video-Frame-Interpolation-b3f639b21ad34b09a72aa2b3325da9f3)

### Datasets

  * [Vimeo-90k-septuplet](http://toflow.csail.mit.edu/index.html#septuplet)      
  * Adobe240fps      
  * Sintel      
  * UCF101      
  * Davis      
  
### Train & Test

   python train.py --flow_model --depth_model --data_root --out_root --softsplat      
   python test.py --flow_model --depth_model --data_root --out_root --softsplat      
   python test_without_refine.py --flow_model --depth_model --data_root --out_root --softsplat      

## 🎮 Dynamic Object Semantic Segmentation
### To be updated

