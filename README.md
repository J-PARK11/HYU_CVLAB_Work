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
```shell
   # Train
   python train.py --flow_model --depth_model --data_root --out_root --softsplat

   # Test
   python test.py --flow_model --depth_model --data_root --out_root --softsplat
   
   # Test without refinet
   python test_without_refine.py --flow_model --depth_model --data_root --out_root --softsplat      
```

## 🎮 Dynamic Object Semantic Segmentation

### Descipline
* Visual SLAM, Semantic Segmentation 기술을 융합하여 동적 물체가 존재하는 상황에서의 측위 및
지도작성을 더 정확하게 수행할 수 있음.          
* ORB [3] , SIFT [4] 등의 feature extractor로 이미지의 feature를 추출후, Semantic Segmentation 결과 중 동적물체 mask에 해당하는 feature를 제거.            
* 이후 남아있는 정적인 feature들을 이용하여 visual odometry 진행함으로 동적환경에서도 강인하게 동작하는 visual SLAM구현가능            

### Yolov8
#### Config
```
    Train Dataset: COCO Dataset (80 class)            
    Dynamic & Static Class: 19 vs 61 class            
    Infernce Duration per image: i.1ms preprocess + 3.0ms inference + 1.1ms postprocess = 5.2ms            
    Test Inference Image Size: (480, 640)            
    Test GPU: RTX 2080Ti            
```

### Future Work
```
    C 개발 환경 포팅.            
    so파일 구축하여 C 호환 여부 확인.            
    ORB SLAM에 이식하여 Dynamic Object Feature Removal            
```

