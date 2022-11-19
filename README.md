# TensorflowLiteCplusplus
 TensorflowLiteCplusplus Linux_x64
 
 `sudo cp -r include /usr/local`
 
 `sudo cp -r libtensorflowlite.so  /usr/local/lib`
 
 `sudo ldconfig /usr/local/lib`
 
 ```
 g++ MobileNetV1LX64.cpp -o MobileNetV1LX64 -ltensorflowlite `pkg-config --cflags --libs opencv4`
 ```
 
 
