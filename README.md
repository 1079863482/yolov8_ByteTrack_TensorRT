# yolov8_ByteTrack_TensorRT 

yolov8 ByteTrack TensorRT C++ 实现

## 使用

```python
git clone https://github.com/1079863482/yolov8_ByteTrack_TensorRT.git

cd yolov8_ByteTrack_TensorRT

```

下载并编译eigen3，[密码ueq4](https://pan.baidu.com/s/15kEfCxpy-T7tz60msxxExg)

```python
unzip eigen-3.3.9.zip
cd eigen-3.3.9
mkdir build
cd build
cmake ..
sudo make install
```


修改CMakeList.txt，主要是一下几个方面

```python
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.4/bin/nvcc)          # cuda版本

# TensorRT
set(TensorRT_INCLUDE_DIRS /home/cai/TensorRT-8.5.1.7/include)   # trt路径
set(TensorRT_LIBRARIES /home/cai/TensorRT-8.5.1.7/lib) 


list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ./include
        /usr/local/include/eigen3                # 检查路径是否正确
        )

```


编译：

```python
mkdir build
cd build
cmake ..
make -j8
```

运行：
```python
./yolov8 ./model/yolov8n.engine data/4.mp4
```



