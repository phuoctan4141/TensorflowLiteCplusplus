# TensorflowLiteCplusplus
 TensorflowLiteCplusplus Linux_x64
 
bazel build -c opt //tensorflow/lite:libtensorflowlite.so

bazel build -c opt //tensorflow/lite/experimental/c:libtensorflowlite_c.so

[find ./lite -name "*.h" | tar -cf headers.tar -T -](https://www.thecodingnotebook.com/2019/11/cross-platform-object-detection-with.html)
 
 `sudo cp -r include /usr/local`
 
 `sudo cp -r libtensorflowlite.so  /usr/local/lib`
 
 `sudo ldconfig /usr/local/lib`
 
 ```
 g++ MobileNetV1LX64.cpp -o MobileNetV1LX64 -ltensorflowlite `pkg-config --cflags --libs opencv4`
 ```
 
 
